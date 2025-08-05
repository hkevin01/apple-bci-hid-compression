import Foundation
import HIDInterface

public struct NeuralData {
    public let timestamp: TimeInterval
    public let channels: [String]
    public let samples: [[Double]]
    
    public init(timestamp: TimeInterval = Date().timeIntervalSince1970,
                channels: [String],
                samples: [[Double]]) {
        self.timestamp = timestamp
        self.channels = channels
        self.samples = samples
    }
}

public enum CompressionQuality {
    case lossless
    case lossy(compressionRatio: Double)
}

public protocol NeuralCompressor {
    func compress(_ data: NeuralData, quality: CompressionQuality) async throws -> Data
    func decompress(_ data: Data) async throws -> NeuralData
}

public actor BCIController {
    private let compressor: any NeuralCompressor
    private let hidInterface: HIDInterface
    
    public init(compressor: any NeuralCompressor, hidInterface: HIDInterface) {
        self.compressor = compressor
        self.hidInterface = hidInterface
    }
    
    public func processNeuralInput(_ data: NeuralData) async throws -> HIDEvent {
        let compressed = try await compressor.compress(data, quality: .lossy(compressionRatio: 0.1))
        let decompressed = try await compressor.decompress(compressed)
        return try await hidInterface.translateNeuralData(decompressed)
    }
}
