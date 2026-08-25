import Darwin
import Foundation
import Metal

private let dispatchKind = "open_cake_metal_operator_dispatch_v1"
private let tokenCount = 8
private let expertCount = 4
private let rowCount = 8
private let routeCount = 8
private let featureCount = 16

private enum ProbeError: Error, CustomStringConvertible {
    case failed(String)

    var description: String {
        switch self {
        case .failed(let message):
            return message
        }
    }
}

private struct Launch {
    let kind: String
    let entryPoint: String
    let bufferOrder: [String]
    let threadgroupsPerGrid: [Int]
    let threadsPerThreadgroup: [Int]
    let threadgroupMemoryBytes: Int

    func bufferIndex(_ name: String) throws -> Int {
        guard let index = bufferOrder.firstIndex(of: name) else {
            throw ProbeError.failed("\(kind) launch does not bind buffer \(name)")
        }
        return index
    }
}

private func require<T>(_ value: T?, _ message: String) throws -> T {
    guard let value else {
        throw ProbeError.failed(message)
    }
    return value
}

private func object(_ value: Any?, _ context: String) throws -> [String: Any] {
    guard let result = value as? [String: Any] else {
        throw ProbeError.failed("\(context) must be an object")
    }
    return result
}

private func string(_ value: Any?, _ context: String) throws -> String {
    guard let result = value as? String, !result.isEmpty else {
        throw ProbeError.failed("\(context) must be a non-empty string")
    }
    return result
}

private func integer(_ value: Any?, _ context: String, minimum: Int = 0) throws -> Int {
    guard let number = value as? NSNumber,
          CFGetTypeID(number) != CFBooleanGetTypeID() else {
        throw ProbeError.failed("\(context) must be an integer")
    }
    let result = number.intValue
    guard NSNumber(value: result) == number, result >= minimum else {
        throw ProbeError.failed("\(context) is outside the admitted integer range")
    }
    return result
}

private func stringArray(_ value: Any?, _ context: String) throws -> [String] {
    guard let values = value as? [Any] else {
        throw ProbeError.failed("\(context) must be an array")
    }
    return try values.enumerated().map { index, value in
        try string(value, "\(context)[\(index)]")
    }
}

private func dimensions(_ value: Any?, _ context: String, minimum: Int) throws -> [Int] {
    guard let values = value as? [Any], values.count == 3 else {
        throw ProbeError.failed("\(context) must contain three dimensions")
    }
    return try values.enumerated().map { index, value in
        try integer(value, "\(context)[\(index)]", minimum: minimum)
    }
}

private func parseLaunches(path: String) throws -> [Launch] {
    let url = URL(fileURLWithPath: path)
    let document = try object(
        try JSONSerialization.jsonObject(with: Data(contentsOf: url)),
        "launch document"
    )
    guard try integer(document["schema_version"], "schema_version", minimum: 1) == 1,
          let values = document["operators"] as? [Any],
          values.count == 2 else {
        throw ProbeError.failed("launch document route contract differs")
    }
    let launches = try values.enumerated().map { index, value -> Launch in
        let item = try object(value, "operators[\(index)]")
        return Launch(
            kind: try string(item["kind"], "operators[\(index)].kind"),
            entryPoint: try string(
                item["entry_point"], "operators[\(index)].entry_point"
            ),
            bufferOrder: try stringArray(
                item["buffer_order"], "operators[\(index)].buffer_order"
            ),
            threadgroupsPerGrid: try dimensions(
                item["threadgroups_per_grid"],
                "operators[\(index)].threadgroups_per_grid",
                minimum: 1
            ),
            threadsPerThreadgroup: try dimensions(
                item["threads_per_threadgroup"],
                "operators[\(index)].threads_per_threadgroup",
                minimum: 1
            ),
            threadgroupMemoryBytes: try integer(
                item["threadgroup_memory_bytes"],
                "operators[\(index)].threadgroup_memory_bytes"
            )
        )
    }
    guard launches.map(\.kind) == ["indexed_gather", "weighted_combine"] else {
        throw ProbeError.failed("launch document must name the two admitted operator kinds")
    }
    return launches
}

// Round-to-nearest, ties-to-even conversion. The probe only supplies finite inputs, but
// retaining the NaN payload rule keeps this helper a correct BF16 conversion in isolation.
private func floatToBFloat16RNE(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    if (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0 {
        return UInt16((bits >> 16) | 0x0040)
    }
    let roundingBias = UInt32(0x7fff) + ((bits >> 16) & 1)
    return UInt16((bits &+ roundingBias) >> 16)
}

private func bfloat16ToFloat(_ bits: UInt16) -> Float {
    Float(bitPattern: UInt32(bits) << 16)
}

private func makeBuffer<T>(
    _ device: MTLDevice,
    values: [T],
    label: String
) throws -> MTLBuffer {
    guard !values.isEmpty else {
        throw ProbeError.failed("\(label) values must not be empty")
    }
    return try values.withUnsafeBytes { bytes in
        try require(
            device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: bytes.count,
                options: .storageModeShared
            ),
            "failed to allocate \(label)"
        )
    }
}

private func makeOutputBuffer(
    _ device: MTLDevice,
    elementCount: Int,
    label: String
) throws -> MTLBuffer {
    let byteCount = elementCount * MemoryLayout<UInt16>.stride
    let buffer = try require(
        device.makeBuffer(length: byteCount, options: .storageModeShared),
        "failed to allocate \(label)"
    )
    memset(buffer.contents(), 0xa5, byteCount)
    return buffer
}

private func readBFloat16Bits(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
    let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}

private func commandStatus(_ status: MTLCommandBufferStatus) -> String {
    switch status {
    case .notEnqueued: return "not_enqueued"
    case .enqueued: return "enqueued"
    case .committed: return "committed"
    case .scheduled: return "scheduled"
    case .completed: return "completed"
    case .error: return "error"
    @unknown default: return "unknown"
    }
}

private func pipelineObservation(
    _ pipeline: MTLComputePipelineState,
    launch: Launch
) -> [String: Any] {
    [
        "entry_point": launch.entryPoint,
        "thread_execution_width": pipeline.threadExecutionWidth,
        "max_total_threads_per_threadgroup": pipeline.maxTotalThreadsPerThreadgroup,
        "static_threadgroup_memory_length": pipeline.staticThreadgroupMemoryLength,
        "threadgroups_per_grid": launch.threadgroupsPerGrid,
        "threads_per_threadgroup": launch.threadsPerThreadgroup,
        "threadgroup_memory_bytes": launch.threadgroupMemoryBytes,
    ]
}

private func validatePipeline(
    _ pipeline: MTLComputePipelineState,
    launch: Launch
) throws {
    let requested = launch.threadsPerThreadgroup.reduce(1, *)
    guard pipeline.threadExecutionWidth == 32 else {
        throw ProbeError.failed(
            "\(launch.kind) requires a 32-thread SIMDgroup, "
                + "observed \(pipeline.threadExecutionWidth)"
        )
    }
    guard launch.threadsPerThreadgroup[1] == 1,
          launch.threadsPerThreadgroup[2] == 1,
          requested % pipeline.threadExecutionWidth == 0,
          requested <= pipeline.maxTotalThreadsPerThreadgroup else {
        throw ProbeError.failed("\(launch.kind) lowering launch is not admitted by its pipeline")
    }
    guard launch.threadgroupMemoryBytes >= pipeline.staticThreadgroupMemoryLength else {
        throw ProbeError.failed(
            "\(launch.kind) lowering under-declares static threadgroup memory"
        )
    }
}

private func dispatch(
    device: MTLDevice,
    queue: MTLCommandQueue,
    library: MTLLibrary,
    launch: Launch,
    buffers: [String: MTLBuffer]
) throws -> (MTLComputePipelineState, String) {
    let function = try require(
        library.makeFunction(name: launch.entryPoint),
        "metallib entry point \(launch.entryPoint) is missing"
    )
    let pipeline = try device.makeComputePipelineState(function: function)
    try validatePipeline(pipeline, launch: launch)
    let commandBuffer = try require(queue.makeCommandBuffer(), "failed to create command buffer")
    let encoder = try require(
        commandBuffer.makeComputeCommandEncoder(),
        "failed to create compute command encoder"
    )
    encoder.setComputePipelineState(pipeline)
    for (name, buffer) in buffers {
        encoder.setBuffer(buffer, offset: 0, index: try launch.bufferIndex(name))
    }
    if launch.threadgroupMemoryBytes > 0 {
        encoder.setThreadgroupMemoryLength(launch.threadgroupMemoryBytes, index: 0)
    }
    encoder.dispatchThreadgroups(
        MTLSize(
            width: launch.threadgroupsPerGrid[0],
            height: launch.threadgroupsPerGrid[1],
            depth: launch.threadgroupsPerGrid[2]
        ),
        threadsPerThreadgroup: MTLSize(
            width: launch.threadsPerThreadgroup[0],
            height: launch.threadsPerThreadgroup[1],
            depth: launch.threadsPerThreadgroup[2]
        )
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        throw ProbeError.failed("\(launch.kind) command buffer failed: \(error)")
    }
    guard commandBuffer.status == .completed else {
        throw ProbeError.failed(
            "\(launch.kind) command buffer did not complete "
                + "(status \(commandBuffer.status.rawValue))"
        )
    }
    return (pipeline, commandStatus(commandBuffer.status))
}

private func runProbe(metallibPath: String, launchPath: String) throws -> [String: Any] {
    let metallibURL = URL(fileURLWithPath: metallibPath)
    guard metallibURL.path.hasPrefix("/") else {
        throw ProbeError.failed("metallib path must be absolute")
    }
    let metallibBytes = try Data(contentsOf: metallibURL, options: .mappedIfSafe)
    guard metallibBytes.count >= 4,
          Array(metallibBytes.prefix(4)) == [0x4d, 0x54, 0x4c, 0x42] else {
        throw ProbeError.failed("metallib is missing the MTLB header")
    }
    let launches = try parseLaunches(path: launchPath)
    let device = try require(MTLCreateSystemDefaultDevice(), "Metal device is unavailable")
    guard device.supportsFamily(.apple9) else {
        throw ProbeError.failed("Metal device does not support Apple GPU family 9")
    }
    let library = try device.makeLibrary(URL: metallibURL)
    let queue = try require(device.makeCommandQueue(), "failed to create command queue")

    var expertRows = [UInt16]()
    expertRows.reserveCapacity(expertCount * rowCount * featureCount)
    for expert in 0..<expertCount {
        for row in 0..<rowCount {
            for feature in 0..<featureCount {
                let numerator = Float(expert * 128 + row * 16 + feature - 256)
                expertRows.append(floatToBFloat16RNE(numerator / 32.0))
            }
        }
    }

    var expertIDs = [Int32]()
    var rowIDs = [Int32]()
    var routeWeights = [Float]()
    // Route 7's exact Float value makes two final accumulators land exactly halfway
    // between adjacent BF16 values: one with an even retained LSB and one with an odd
    // retained LSB. The output comparison therefore distinguishes ties-to-even from
    // truncation and from a one-direction halfway rule.
    let fixedWeights: [Float] = [
        0.5, -0.25, 0.125, -0.0625, 0.3, -0.2, 0.1, 0.04277785122394562,
    ]
    for token in 0..<tokenCount {
        for route in 0..<routeCount {
            switch route {
            case 0:
                expertIDs.append(-1)
                rowIDs.append(Int32((token + route) % rowCount))
            case 1:
                expertIDs.append(Int32(expertCount))
                rowIDs.append(Int32((token + route) % rowCount))
            case 2:
                expertIDs.append(Int32((token + route) % expertCount))
                rowIDs.append(-1)
            case 3:
                expertIDs.append(Int32((token + route) % expertCount))
                rowIDs.append(Int32(rowCount))
            default:
                expertIDs.append(Int32((token + route) % expertCount))
                rowIDs.append(Int32((token * 3 + route) % rowCount))
            }
            routeWeights.append(fixedWeights[route])
        }
    }

    let validPairs = zip(expertIDs, rowIDs).filter { expert, row in
        expert >= 0 && expert < Int32(expertCount) && row >= 0 && row < Int32(rowCount)
    }.count
    let inputCoverage: [String: Any] = [
        "valid_pairs": validPairs,
        "negative_expert_ids": expertIDs.filter { $0 == -1 }.count,
        "upper_bound_expert_ids": expertIDs.filter { $0 == Int32(expertCount) }.count,
        "negative_row_ids": rowIDs.filter { $0 == -1 }.count,
        "upper_bound_row_ids": rowIDs.filter { $0 == Int32(rowCount) }.count,
    ]

    var expectedGather = [UInt16]()
    expectedGather.reserveCapacity(tokenCount * routeCount * featureCount)
    for token in 0..<tokenCount {
        for route in 0..<routeCount {
            let offset = token * routeCount + route
            let expert = Int(expertIDs[offset])
            let row = Int(rowIDs[offset])
            for feature in 0..<featureCount {
                if expert >= 0 && expert < expertCount && row >= 0 && row < rowCount {
                    expectedGather.append(
                        expertRows[(expert * rowCount + row) * featureCount + feature]
                    )
                } else {
                    expectedGather.append(0)
                }
            }
        }
    }

    let expertBuffer = try makeBuffer(device, values: expertRows, label: "expert_rows")
    let expertIDBuffer = try makeBuffer(device, values: expertIDs, label: "expert_ids")
    let rowIDBuffer = try makeBuffer(device, values: rowIDs, label: "row_ids")
    let gatherOutput = try makeOutputBuffer(
        device,
        elementCount: expectedGather.count,
        label: "gathered_rows"
    )
    let gatherLaunch = launches[0]
    let (gatherPipeline, gatherStatus) = try dispatch(
        device: device,
        queue: queue,
        library: library,
        launch: gatherLaunch,
        buffers: [
            "expert_rows": expertBuffer,
            "expert_ids": expertIDBuffer,
            "row_ids": rowIDBuffer,
            "gathered_rows": gatherOutput,
        ]
    )
    let observedGather = readBFloat16Bits(gatherOutput, count: expectedGather.count)
    let gatherMismatch = zip(observedGather, expectedGather).filter { $0 != $1 }.count
    guard gatherMismatch == 0 else {
        throw ProbeError.failed("indexed_gather BF16 bit mismatch count: \(gatherMismatch)")
    }

    var expectedCombine = [UInt16]()
    expectedCombine.reserveCapacity(tokenCount * featureCount)
    var exactHalfwayEvenLSB = 0
    var exactHalfwayOddLSB = 0
    for token in 0..<tokenCount {
        for feature in 0..<featureCount {
            var combined = Float(0)
            // This is the Workload Contract's fixed route order. Multiplication is
            // materialized before addition so CPU contraction cannot change the oracle.
            for route in 0..<routeCount {
                let offset = token * routeCount + route
                let expert = Int(expertIDs[offset])
                let row = Int(rowIDs[offset])
                let selected: Float
                if expert >= 0 && expert < expertCount && row >= 0 && row < rowCount {
                    selected = bfloat16ToFloat(
                        expertRows[(expert * rowCount + row) * featureCount + feature]
                    )
                } else {
                    selected = 0
                }
                let weighted = selected * routeWeights[offset]
                combined = combined + weighted
            }
            if combined.bitPattern & 0xffff == 0x8000 {
                if (combined.bitPattern >> 16) & 1 == 0 {
                    exactHalfwayEvenLSB += 1
                } else {
                    exactHalfwayOddLSB += 1
                }
            }
            expectedCombine.append(floatToBFloat16RNE(combined))
        }
    }
    guard exactHalfwayEvenLSB == 1, exactHalfwayOddLSB == 1 else {
        throw ProbeError.failed(
            "weighted_combine oracle no longer covers both BF16 halfway parities"
        )
    }

    let weightBuffer = try makeBuffer(
        device,
        values: routeWeights,
        label: "route_weights"
    )
    let combineOutput = try makeOutputBuffer(
        device,
        elementCount: expectedCombine.count,
        label: "output"
    )
    let combineLaunch = launches[1]
    let (combinePipeline, combineStatus) = try dispatch(
        device: device,
        queue: queue,
        library: library,
        launch: combineLaunch,
        buffers: [
            "expert_rows": expertBuffer,
            "expert_ids": expertIDBuffer,
            "row_ids": rowIDBuffer,
            "route_weights": weightBuffer,
            "output": combineOutput,
        ]
    )
    let observedCombine = readBFloat16Bits(combineOutput, count: expectedCombine.count)
    let combineMismatch = zip(observedCombine, expectedCombine).filter { $0 != $1 }.count
    guard combineMismatch == 0 else {
        throw ProbeError.failed("weighted_combine BF16 bit mismatch count: \(combineMismatch)")
    }

    return [
        "schema_version": 1,
        "kind": dispatchKind,
        "status": "passed",
        "kernel_calls": 2,
        "device": [
            "name": device.name,
            "registry_id": device.registryID,
            "supports_apple9": true,
            "has_unified_memory": device.hasUnifiedMemory,
            "max_threadgroup_memory_length": device.maxThreadgroupMemoryLength,
        ],
        "input_coverage": inputCoverage,
        "operators": [
            [
                "kind": gatherLaunch.kind,
                "entry_point": gatherLaunch.entryPoint,
                "kernel_calls": 1,
                "pipeline": pipelineObservation(gatherPipeline, launch: gatherLaunch),
                "result": [
                    "output_element_count": observedGather.count,
                    "expected_bf16_bits": expectedGather,
                    "observed_bf16_bits": observedGather,
                    "mismatch_count": gatherMismatch,
                    "all_bits_match": true,
                    "command_buffer_status": gatherStatus,
                ],
            ],
            [
                "kind": combineLaunch.kind,
                "entry_point": combineLaunch.entryPoint,
                "kernel_calls": 1,
                "pipeline": pipelineObservation(combinePipeline, launch: combineLaunch),
                "result": [
                    "output_element_count": observedCombine.count,
                    "expected_bf16_bits": expectedCombine,
                    "observed_bf16_bits": observedCombine,
                    "mismatch_count": combineMismatch,
                    "all_bits_match": true,
                    "command_buffer_status": combineStatus,
                    "cpu_oracle_accumulation_order": "route_0_through_7_fp32",
                    "output_conversion": "bfloat16_round_to_nearest_ties_to_even",
                    "bf16_exact_halfway_cases": exactHalfwayEvenLSB + exactHalfwayOddLSB,
                    "bf16_exact_halfway_even_lsb_cases": exactHalfwayEvenLSB,
                    "bf16_exact_halfway_odd_lsb_cases": exactHalfwayOddLSB,
                ],
            ],
        ],
    ]
}

do {
    guard CommandLine.arguments.count == 3 else {
        throw ProbeError.failed(
            "usage: run_metal_operators.swift METALLIB LOWERING_LAUNCH_JSON"
        )
    }
    let observation = try runProbe(
        metallibPath: CommandLine.arguments[1],
        launchPath: CommandLine.arguments[2]
    )
    guard JSONSerialization.isValidJSONObject(observation) else {
        throw ProbeError.failed("operator observation is not valid JSON")
    }
    let encoded = try JSONSerialization.data(
        withJSONObject: observation,
        options: [.sortedKeys]
    )
    FileHandle.standardOutput.write(encoded)
    FileHandle.standardOutput.write(Data([0x0a]))
} catch {
    let message = "metal operator probe failed: \(error)\n"
    FileHandle.standardError.write(message.data(using: .utf8)!)
    exit(1)
}
