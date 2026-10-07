"""Native-MLX component/state replay using the same inputs as CoreAI."""
import numpy as np

from recipes._validation import compare, compare_cache, require_full_precision_mlx
from .adapter import StreamingDecoder
from .build import load_source


class Reference:
    def __init__(self, bundle, *, source=None):
        require_full_precision_mlx("Pocket TTS")
        self.model, _ = load_source(source or bundle.metadata["source"])
        self.blocks = bundle.metadata["backbone_components"]
        self.steps = bundle.metadata["flow_steps"]
        self.stride = bundle.metadata["decoder_steps_per_frame"]
        self.decoder = StreamingDecoder(self.model)
        self.checks = {}
        self.offset = 0

    def snapshot_backbone(self, session):
        states = [session.snapshot_state(name) for name in self.blocks]
        return {key: np.concatenate([state[key] for state in states]) for key in states[0]}

    def __call__(self, session, component, inputs, outputs):
        import mlx.core as mx
        from mlx_audio.tts.models.pocket_tts.conditioners import TokenizedText
        from mlx_audio.tts.models.pocket_tts.flow_lm import lsd_decode

        flow = self.model.flow_lm
        actual = {name: value.numpy().copy() for name, value in outputs.items()}
        if component == "conditioner":
            self.caches = flow.make_cache()
            self.model.mimi.reset_state()
            self.offset = 0
            compare(self.checks, "conditioner", actual["embeddings"],
                    np.asarray(flow.conditioner(TokenizedText(mx.array(inputs["tokens"])))))
        if component == self.blocks[0]:
            self.embeddings = np.asarray(inputs["embeddings"]).copy()
        if component == self.blocks[-1]:
            reference = flow.out_norm(flow.transformer(mx.array(self.embeddings), self.caches))
            self.offset += self.embeddings.shape[1]
            compare(self.checks, "backbone.hidden", actual["hidden"], np.asarray(reference))
            compare(self.checks, "backbone.eos", actual["eos"], np.asarray(flow.out_eos(reference)))
            compare_cache(self.checks, "backbone", self.snapshot_backbone(session), self.caches, self.offset)
        if component == "sampler":
            hidden, noise = mx.array(inputs["hidden"]), mx.array(inputs["noise"])
            reference = lsd_decode(lambda s, t, x: flow.flow_net(hidden, s, t, x), noise, self.steps)
            compare(self.checks, "sampler.latent", actual["latent"], np.asarray(reference))
            compare(self.checks, "sampler.embedding", actual["embedding"], np.asarray(flow.input_linear(reference[:, None])))
        if component == "decoder":
            value = (mx.array(inputs["latent"]) * flow.emb_std + flow.emb_mean).transpose(0, 2, 1)
            reference = self.model.mimi.decode_step(self.model.mimi.quantizer(value))
            compare(self.checks, "decoder.audio", actual["audio"], np.asarray(reference))
            compare_cache(self.checks, "backbone.after_decoder", self.snapshot_backbone(session), self.caches, self.offset)
            state = session.snapshot_state("decoder")
            position = int(inputs["position"][0]) + inputs["latent"].shape[1] * self.stride
            compare_cache(self.checks, "decoder", state, self.model.mimi.decoder_cache, position)
            for module, attr, spec in self.decoder.convolutions:
                reference = getattr(module, attr)
                if attr == "_prev_ys" and module.convtr.convtr.bias is not None:
                    reference = reference - module.convtr.convtr.bias[None, :, None]
                compare(self.checks, "decoder." + spec.name, state[spec.name], np.asarray(reference))
