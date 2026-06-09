import CoreAI
import Darwin
import Foundation

struct BenchmarkOptions {
    var assetPath: String
    var contexts: [Int] = [16, 32, 64, 128, 256]
    var steps: Int = 16
    var warmup: Int = 1
    var functionName = "main"
    var inputName = "input_ids"
    var positionIdsName = "position_ids"
    var outputName: String?
    var fillTokenId: Int32 = 0
    var growContext = false

    static func parse(_ arguments: ArraySlice<String>) throws -> BenchmarkOptions {
        guard let assetPath = arguments.first else {
            fputs(usage, stderr)
            exit(2)
        }
        var options = BenchmarkOptions(assetPath: assetPath)
        var index = arguments.index(after: arguments.startIndex)
        while index < arguments.endIndex {
            let flag = arguments[index]
            switch flag {
            case "--contexts":
                options.contexts = try parseContexts(requireValue(arguments, after: &index, flag: flag))
            case "--steps":
                options.steps = try parsePositiveInt(requireValue(arguments, after: &index, flag: flag), flag: flag)
            case "--warmup":
                options.warmup = try parseNonNegativeInt(requireValue(arguments, after: &index, flag: flag), flag: flag)
            case "--function-name":
                options.functionName = requireValue(arguments, after: &index, flag: flag)
            case "--input-name":
                options.inputName = requireValue(arguments, after: &index, flag: flag)
            case "--position-ids-name":
                options.positionIdsName = requireValue(arguments, after: &index, flag: flag)
            case "--output-name":
                options.outputName = requireValue(arguments, after: &index, flag: flag)
            case "--fill-token-id":
                let value = try parseNonNegativeInt(requireValue(arguments, after: &index, flag: flag), flag: flag)
                options.fillTokenId = Int32(value)
            case "--grow-context":
                options.growContext = true
            default:
                throw BackendError.invalidArgument("unknown Swift backend argument: \(flag)")
            }
            index = arguments.index(after: index)
        }
        return options
    }

    private static let usage = """
        usage: benchmark_aimodel_sampling_coreai <model.aimodel> [--contexts 16,32] [--steps N]

        """

    private static func requireValue(
        _ arguments: ArraySlice<String>,
        after index: inout ArraySlice<String>.Index,
        flag: String
    ) -> String {
        let valueIndex = arguments.index(after: index)
        guard valueIndex < arguments.endIndex else {
            fputs("missing value for \(flag)\n", stderr)
            exit(2)
        }
        index = valueIndex
        return arguments[valueIndex]
    }

    private static func parseContexts(_ value: String) throws -> [Int] {
        let contexts = try value.split(separator: ",").map {
            try parsePositiveInt(String($0), flag: "--contexts")
        }
        if contexts.isEmpty {
            throw BackendError.invalidArgument("--contexts must contain at least one positive integer")
        }
        return contexts
    }

    private static func parsePositiveInt(_ value: String, flag: String) throws -> Int {
        guard let parsed = Int(value), parsed > 0 else {
            throw BackendError.invalidArgument("\(flag) must be a positive integer, got \(value)")
        }
        return parsed
    }

    private static func parseNonNegativeInt(_ value: String, flag: String) throws -> Int {
        guard let parsed = Int(value), parsed >= 0 else {
            throw BackendError.invalidArgument("\(flag) must be non-negative, got \(value)")
        }
        return parsed
    }
}

@main
struct CoreAIBenchmarkBackend {
    static func main() async {
        do {
            try await run()
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    static func run() async throws {
        let runOptions = try BenchmarkOptions.parse(CommandLine.arguments.dropFirst())

        let modelURL = URL(fileURLWithPath: runOptions.assetPath)

        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = false
        let model = try await AIModel(contentsOf: modelURL, options: options)
        guard let descriptor = model.functionDescriptor(for: runOptions.functionName) else {
            throw BackendError.missingFunction(runOptions.functionName)
        }
        guard let function = try model.loadFunction(named: runOptions.functionName) else {
            throw BackendError.missingFunction(runOptions.functionName)
        }
        guard descriptor.inputNames.count == 2 else {
            throw BackendError.invalidModel("expected 2 inputs, got \(descriptor.inputNames)")
        }
        guard descriptor.stateNames.count == 2 else {
            throw BackendError.invalidModel("expected 2 states, got \(descriptor.stateNames)")
        }
        guard let outputName = runOptions.outputName ?? descriptor.outputNames.first else {
            throw BackendError.invalidModel("expected at least one output")
        }

        let inputName = runOptions.inputName
        let positionName = runOptions.positionIdsName
        let keyName = descriptor.stateNames[0]
        let valueName = descriptor.stateNames[1]

        let inputDesc = try ndArrayDescriptor(descriptor.inputDescriptor(of: inputName), name: inputName)
        let positionDesc = try ndArrayDescriptor(descriptor.inputDescriptor(of: positionName), name: positionName)
        let keyDesc = try ndArrayDescriptor(descriptor.stateDescriptor(of: keyName), name: keyName)
        let valueDesc = try ndArrayDescriptor(descriptor.stateDescriptor(of: valueName), name: valueName)
        let logitsDesc = try ndArrayDescriptor(descriptor.outputDescriptor(of: outputName), name: outputName)
        let vocabSize = logitsDesc.shape.last ?? 0

        printTableHeader()

        for contextLength in runOptions.contexts {
            let stateCapacity = contextLength + (runOptions.growContext ? runOptions.steps + 1 : 1)
            var keyCache = NDArray(descriptor: keyDesc.resolvingDynamicDimensions(
                keyDesc.shape.map { $0 < 0 ? stateCapacity : $0 }))
            var valueCache = NDArray(descriptor: valueDesc.resolvingDynamicDimensions(
                valueDesc.shape.map { $0 < 0 ? stateCapacity : $0 }))
            var token = runOptions.fillTokenId
            var position = contextLength

            token = try await runBatch(
                function: function,
                inputName: inputName,
                positionName: positionName,
                outputName: outputName,
                inputDesc: inputDesc,
                positionDesc: positionDesc,
                logitsDesc: logitsDesc,
                keyName: keyName,
                valueName: valueName,
                keyCache: &keyCache,
                valueCache: &valueCache,
                tokens: Array(repeating: token, count: contextLength),
                totalPositions: contextLength,
                vocabSize: vocabSize
            )

            for _ in 0..<runOptions.warmup {
                token = try await runBatch(
                    function: function,
                    inputName: inputName,
                    positionName: positionName,
                    outputName: outputName,
                    inputDesc: inputDesc,
                    positionDesc: positionDesc,
                    logitsDesc: logitsDesc,
                    keyName: keyName,
                    valueName: valueName,
                    keyCache: &keyCache,
                    valueCache: &valueCache,
                    tokens: [token],
                    totalPositions: position + 1,
                    vocabSize: vocabSize
                )
            }

            let startPosition = position
            let start = Date()
            for _ in 0..<runOptions.steps {
                token = try await runBatch(
                    function: function,
                    inputName: inputName,
                    positionName: positionName,
                    outputName: outputName,
                    inputDesc: inputDesc,
                    positionDesc: positionDesc,
                    logitsDesc: logitsDesc,
                    keyName: keyName,
                    valueName: valueName,
                    keyCache: &keyCache,
                    valueCache: &valueCache,
                    tokens: [token],
                    totalPositions: position + 1,
                    vocabSize: vocabSize
                )
                if runOptions.growContext {
                    position += 1
                }
            }
            let elapsed = Date().timeIntervalSince(start)
            printTableRow(
                contextLength: contextLength,
                steps: runOptions.steps,
                elapsed: elapsed,
                outputName: outputName,
                startPosition: startPosition,
                endPosition: position
            )
        }
    }

    static func runBatch(
        function: InferenceFunction,
        inputName: String,
        positionName: String,
        outputName: String,
        inputDesc: NDArrayDescriptor,
        positionDesc: NDArrayDescriptor,
        logitsDesc: NDArrayDescriptor,
        keyName: String,
        valueName: String,
        keyCache: inout NDArray,
        valueCache: inout NDArray,
        tokens: [Int32],
        totalPositions: Int,
        vocabSize: Int
    ) async throws -> Int32 {
        var inputIds = NDArray(descriptor: inputDesc.resolvingDynamicDimensions([1, tokens.count]))
        fillInt32(&inputIds, values: tokens)

        var positionIds = NDArray(descriptor: positionDesc.resolvingDynamicDimensions([1, totalPositions]))
        fillInt32(&positionIds, values: (0..<totalPositions).map { Int32($0) })

        var logits = NDArray(descriptor: logitsDesc.resolvingDynamicDimensions([1, tokens.count, vocabSize]))

        var states = InferenceFunction.MutableViews()
        states.insert(&keyCache, for: keyName)
        states.insert(&valueCache, for: valueName)

        var outputs = InferenceFunction.MutableViews()
        outputs.insert(&logits, for: outputName)

        _ = try await function.run(
            inputs: [inputName: inputIds, positionName: positionIds],
            states: consume states,
            outputViews: consume outputs
        )
        return greedyToken(logits: logits, tokenCount: tokens.count, vocabSize: vocabSize)
    }
}

enum BackendError: Error, CustomStringConvertible {
    case invalidArgument(String)
    case invalidModel(String)
    case missingFunction(String)
    case missingDescriptor(String)

    var description: String {
        switch self {
        case .invalidArgument(let message): message
        case .invalidModel(let message): message
        case .missingFunction(let name): "missing function \(name)"
        case .missingDescriptor(let name): "missing NDArray descriptor for \(name)"
        }
    }
}

func ndArrayDescriptor(_ descriptor: InferenceValue.Descriptor?, name: String) throws -> NDArrayDescriptor {
    guard case .ndArray(let ndArrayDescriptor) = descriptor else {
        throw BackendError.missingDescriptor(name)
    }
    return ndArrayDescriptor
}

func fillInt32(_ array: inout NDArray, values: [Int32]) {
    var view = array.mutableView(as: Int32.self)
    view.withUnsafeMutablePointer { pointer, _, _ in
        for i in values.indices {
            pointer[i] = values[i]
        }
    }
}

func greedyToken(logits: NDArray, tokenCount: Int, vocabSize: Int) -> Int32 {
    let offset = max(0, tokenCount - 1) * vocabSize
    let view = logits.view(as: Float16.self)
    var bestIndex = 0
    var bestValue = -Float.infinity
    view.withUnsafePointer { pointer, _, _ in
        for i in 0..<vocabSize {
            let value = Float(pointer[offset + i])
            if !value.isNaN && value > bestValue {
                bestValue = value
                bestIndex = i
            }
        }
    }
    return Int32(bestIndex)
}

func printTableHeader() {
    print(
        "\(leftPad("context", width: 8)) \(leftPad("steps", width: 6)) \(leftPad("elapsed_s", width: 10)) "
            + "\(leftPad("tok/s", width: 10)) \(leftPad("output", width: 10)) \(leftPad("pos0", width: 8)) "
            + "\(leftPad("pos1", width: 8))"
    )
    print("-------- ------ ---------- ---------- ---------- -------- --------")
}

func printTableRow(
    contextLength: Int,
    steps: Int,
    elapsed: Double,
    outputName: String,
    startPosition: Int,
    endPosition: Int
) {
    let tokensPerSecond = elapsed > 0 ? Double(steps) / elapsed : Double.infinity
    print(
        String(format: "%8d %6d %10.3f %10.2f %@ %8d %8d",
               contextLength,
               steps,
               elapsed,
               tokensPerSecond,
               leftPad(outputName, width: 10),
               startPosition,
               endPosition)
    )
}

func leftPad(_ value: String, width: Int) -> String {
    if value.count >= width {
        return value
    }
    return String(repeating: " ", count: width - value.count) + value
}
