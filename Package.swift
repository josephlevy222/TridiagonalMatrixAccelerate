// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TridiagonalMatrixAccelerate",
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "TridiagonalMatrixAccelerate",
            targets: ["TridiagonalMatrixAccelerate"],
			//dependencies: [.product(name: "Numerics", package: "swift-numerics")]
        ),
    ],
	dependencies: [.package(url: "https://github.com/apple/swift-numerics.git", from: "1.0.0"),
				   // Dependencies declare other packages that this package depends on.
				   // .package(url: /* package url */, from: "1.0.0"),
				  ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "TridiagonalMatrixAccelerate",
			dependencies: [.product(name: "Numerics", package: "swift-numerics")]
        ),
        .testTarget(
            name: "TridiagonalMatrixAccelerateTests",
			dependencies: ["TridiagonalMatrixAccelerate",.product(name: "Numerics", package: "swift-numerics")]
        ),
    ]
)

	
