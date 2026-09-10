"""Compatibility entry point. MLX-LM model policies live in recipes."""
from recipes._mlx_lm.stateful import MLXLMStatefulConversion, convert_mlx_lm_stateful, main, parse_args


def __getattr__(name):
    # Preserve old private helper imports while recipe clients migrate.
    from recipes._mlx_lm import stateful
    return getattr(stateful, name)


if __name__ == "__main__":
    raise SystemExit(main())
