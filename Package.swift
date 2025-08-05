// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "AppleBCIHIDCompression",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .iPadOS(.v16)
    ],
    products: [
        .library(
            name: "AppleBCICore",
            targets: ["AppleBCICore"]
        ),
        .library(
            name: "CompressionBridge",
            targets: ["CompressionBridge"]
        ),
        .library(
            name: "HIDInterface",
            targets: ["HIDInterface"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
        .package(url: "https://github.com/PythonKit/PythonKit", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "AppleBCICore",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics"),
                "HIDInterface"
            ],
            path: "Sources/AppleBCICore"
        ),
        .target(
            name: "CompressionBridge",
            dependencies: [
                "PythonKit"
            ],
            path: "Sources/CompressionBridge"
        ),
        .target(
            name: "HIDInterface",
            dependencies: [],
            path: "Sources/HIDInterface"
        ),
        .testTarget(
            name: "AppleBCICoreTests",
            dependencies: ["AppleBCICore"],
            path: "Tests/AppleBCICore"
        ),
        .testTarget(
            name: "CompressionBridgeTests",
            dependencies: ["CompressionBridge"],
            path: "Tests/CompressionBridge"
        ),
        .testTarget(
            name: "HIDInterfaceTests",
            dependencies: ["HIDInterface"],
            path: "Tests/HIDInterface"
        )
    ]
)
