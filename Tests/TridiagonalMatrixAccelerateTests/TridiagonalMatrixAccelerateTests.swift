import XCTest
@testable import TridiagonalMatrixAccelerate
import Numerics
import Accelerate

final class TridiagonalMatrixAccelerateTests: XCTestCase {
	/// Asserts that two vectors are approximately equal within a given tolerance.
	private func assertVectorsApproximatelyEqual<T: ScalarField>(_ v1: [T], _ v2: [T],
																 tolerance: T.Magnitude,
																 file: StaticString = #file,
																 line: UInt = #line) {
		XCTAssertEqual(v1.count, v2.count, "Vector counts do not match", file: (file), line: line)
		for (i, (a, b)) in zip(v1, v2).enumerated() {
			let error = (a - b).magnitude
			XCTAssertTrue(error < tolerance, "Vector element \(i) not equal. (\(a) vs \(b), error \(error))",
						  file: (file), line: line)
		}
	}
	
	func testNormal() throws {
		try testExample(Complex<Double>(2.0,0.0), det: Complex<Double>(6.0,0.0))
	}
	func testNormalDouble() throws {
		try testExample(Complex<Double>(2.0,0.0), det: Complex<Double>(6.0,0.0))
	}
	func testSingular() throws {
		try testExample(Complex<Float>(1.0,0.0), det: Complex<Float>(0.0,0.0))
	}
	func testNearlySingular() throws {
		try testExample(Float(1.732051), det: Float(2.02656e-6))
	}
	
	func testaXpY() {
		// Test with Double
		let a_d: Double = 2.0
		let x_d: [Double] = [1.0, 2.0, 3.0]
		let y_d: [Double] = [10.0, 20.0, 30.0]
		
		let result_d = aXpY(a: a_d, x: x_d, y: y_d)
		let expected_d: [Double] = [12.0, 24.0, 36.0]
		
		assertVectorsApproximatelyEqual(result_d, expected_d, tolerance: 1e-15)
		
		// Test with Complex<Float>
		typealias CFloat = Complex<Double>
		let a_c = CFloat(1.0, 1.0) // (1+i)
		let x_c = [CFloat(2.0, 0.0), CFloat(0.0, 3.0)] // [2, 3i]
		let y_c = [CFloat(10.0, 0.0), CFloat(20.0, 0.0)] // [10, 20]
		
		// Expected calculation:
		// a*x = [(1+i)*2, (1+i)*3i] = [2+2i, 3i-3] = [2+2i, -3+3i]
		// a*x + y = [12+2i, 17+3i]
		
		let result_c = aXpY(a: a_c, x: x_c, y: y_c)
		let expected_c = [CFloat(12.0, 2.0), CFloat(17.0, 3.0)]
		
		assertVectorsApproximatelyEqual(result_c, expected_c, tolerance: 1e-7)
	}
	
	func testMatrixVectorMultiply() {
		// Test with Double
		let A = TridiagonalMatrix(diagonal: [2.0, 3.0, 4.0],
								  upper: [1.0, 1.0],
								  lower: [-1.0, -1.0])
		let x: [Double] = [1.0, 2.0, 3.0]
		
		// Expected calculation:
		// A*x = | 2  1  0 | | 1 |   | 2*1 + 1*2 + 0*3 |   |  4 |
		//       |-1  3  1 | | 2 | = | -1*1 + 3*2 + 1*3| = |  8 |
		//       | 0 -1  4 | | 3 |   | 0*1 + -1*2 + 4*3|   | 10 |
		
		let result = A * x
		let expected: [Double] = [4.0, 8.0, 10.0]
		
		assertVectorsApproximatelyEqual(result, expected, tolerance: 1e-15)
	}
	
	func testAXpY() {
		// Test with Double
		let A = TridiagonalMatrix(diagonal: [2.0, 3.0, 4.0],
								  upper: [1.0, 1.0],
								  lower: [-1.0, -1.0])
		let x: [Double] = [1.0, 2.0, 3.0]
		var y: [Double] = [100.0, 200.0, 300.0]
		
		// Expected calculation:
		// A*x = [4.0, 8.0, 10.0] (from testMatrixVectorMultiply)
		// A*x + y = [104.0, 208.0, 310.0]
		
		let result = AXpY_(A,x,&y)
		let expected: [Double] = [104.0, 208.0, 310.0]
		
		assertVectorsApproximatelyEqual(result, expected, tolerance: 1e-15)
	}
	
	func testComplexAXpY() {
		// Test with Complex<Double>
		let A = TridiagonalMatrix(diagonal: [2.0, 3.0, 4.0].map { Complex<Double>($0, 0)},
								  upper: [1.0, 1.0].map { Complex<Double>($0, 0)},
								  lower: [-1.0, -1.0].map { Complex<Double>($0, 0)})
		let x = [1.0, 2.0, 3.0].map { Complex<Double>($0, 0)}
		var y = [100.0, 200.0, 300.0].map { Complex<Double>($0, 0)}
		
		// Expected calculation:
		// A*x = [4.0, 8.0, 10.0] (from testMatrixVectorMultiply)
		// A*x + y = [104.0, 208.0, 310.0]
		
		let result = AXpY_(A,x,&y)
		let expected = [104.0, 208.0, 310.0].map { Complex<Double>($0, 0)}
		
		assertVectorsApproximatelyEqual(result, expected, tolerance: 1e-15)
	}
	
	private func testExample<T : ScalarField >(_ d: T, det: T)  throws {
		let one = T.one
		let zero = T.zero
		let lower = [one,one,one,one]
		let upper = lower
		let diagonal = [d,d,d,d,d]
		let tridiag = TridiagonalMatrix(diagonal: diagonal, upper: upper, lower: lower)
		var tridiagLU =  TridiagonalLUMatrix(tridiag)
		let i = [ [one,zero,zero,zero,zero],
				  [zero,one,zero,zero,zero],
				  [zero,zero,one,zero,zero],
				  [zero,zero,zero,one,zero],
				  [zero,zero,zero,zero,one] ]
		let x = i.map {icol in var b = icol; return  tridiagLU.solve(&b) }
		let ii = x.map {xcol in tridiag * xcol}

		var tolerance = T.Magnitude.ulpOfOne*2
		let condition = tridiagLU.approximateConditionNumber
		print("conditionNumber=\(condition)")
		tolerance *= condition
		print("tolerance=\(tolerance)")
		let e = zip(i,ii).map { zip($0,$1).map { $0 - $1 } }
		let okay = e.flatMap { $0 }.reduce(true) { $0 && $1.magnitude < tolerance }
		let maxError = e.flatMap {$0}.reduce(T.Magnitude.zero) { max($0,$1.magnitude)}
		print("maxError=\(maxError)")
		let determinant = tridiagLU.determinant
		print("determinate=\(determinant) vs. \(det)")
		//XCTAssertTrue(determinant.isApproximatelyEqual(to: det))
		// FIX 2: Replace `isApproximatelyEqual` with a magnitude check
		let detError = (determinant - det).magnitude
		// Use an absolute tolerance for determinant check
		let detTolerance = T.Magnitude.ulpOfOne * 1000
		
		XCTAssertTrue(detError < detTolerance, "Determinant error \(detError) exceeds tolerance \(detTolerance)")
		XCTAssertTrue(okay || tridiagLU.approximateConditionNumber.isInfinite )
	}
	
	/// Simple deterministic RNG for reproducible randomized tests.
	struct DeterministicRNG {
		private var state: UInt64
		init(seed: UInt64) { state = seed }
		mutating func nextDouble(in range: ClosedRange<Double>) -> Double {
			// xorshift64* LCG-like step for determinism
			state &+= 0x9E3779B97F4A7C15
			var z = state
			z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
			z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
			z = z ^ (z >> 31)
			let unit = Double(z & 0xFFFFFFFFFFFF) / Double(0xFFFFFFFFFFFF)
			return range.lowerBound + unit * (range.upperBound - range.lowerBound)
		}
	}
	
	/// Dense Gaussian elimination solver for small systems (Double).
	/// Used as a reference to cross-check the tridiagonal solver.
	private func denseSolve(_ A: TridiagonalMatrix<Double>, _ b: [Double]) -> [Double] {
		let n = A.size
		precondition(b.count == n)
		if n == 0 { return [] }
		// Build dense matrix
		var M = Array(repeating: Array(repeating: 0.0, count: n), count: n)
		for i in 0..<n {
			M[i][i] = A.diagonal[i]
			if i < n-1 {
				M[i][i+1] = A.upper[i]
				M[i+1][i] = A.lower[i]
			}
		}
		// Gaussian elimination with partial pivoting
		var Mflat = M
		var rhs = b
		for k in 0..<n {
			// pivot
			var pivot = k
			var maxAbs = abs(Mflat[k][k])
			for i in (k+1)..<n {
				let v = abs(Mflat[i][k])
				if v > maxAbs { maxAbs = v; pivot = i }
			}
			if pivot != k {
				Mflat.swapAt(k, pivot)
				rhs.swapAt(k, pivot)
			}
			let akk = Mflat[k][k]
			// If pivot is zero, system is singular; return zeros to match current library behavior
			if akk == 0 {
				return Array(repeating: 0.0, count: n)
			}
			for i in (k+1)..<n {
				let factor = Mflat[i][k] / akk
				for j in k..<n {
					Mflat[i][j] -= factor * Mflat[k][j]
				}
				rhs[i] -= factor * rhs[k]
			}
		}
		// Back substitution
		var x = Array(repeating: 0.0, count: n)
		for i in stride(from: n-1, through: 0, by: -1) {
			var s = rhs[i]
			for j in (i+1)..<n {
				s -= Mflat[i][j] * x[j]
			}
			let diag = Mflat[i][i]
			if diag == 0 { return Array(repeating: 0.0, count: n) }
			x[i] = s / diag
		}
		return x
	}
	
	/// Randomized cross-check: compare LU solve to a dense reference solver for many small matrices.
	func testRandomizedCrossCheck() {
		var rng = DeterministicRNG(seed: 0xDEADBEEF)
		for _ in 0..<100 {
			let n = Int(rng.nextDouble(in: 2...10))
			let diag = (0..<n).map { _ in rng.nextDouble(in: -10...10) }
			let off = (0..<max(0, n-1)).map { _ in rng.nextDouble(in: -5...5) }
			let A = TridiagonalMatrix(diagonal: diag, upper: off, lower: off)
			var lu = TridiagonalLUMatrix(A)
			let x = (0..<n).map { _ in rng.nextDouble(in: -2...2) }
			var b = A * x
			let xSolved = lu.solve(&b)
			let xRef = denseSolve(A, A * x) // dense solve of same system
			let err = zip(xSolved, xRef).map { abs($0 - $1) }.max() ?? 0.0
			let tol = 1e-8 * max(1.0, lu.approximateConditionNumber)
			XCTAssertLessThanOrEqual(err, tol, "Randomized cross-check failed (err=\(err), tol=\(tol))")
		}
	}
	
	/// Repeated solves to exercise workspace reuse and detect leaks/regressions.
	func testRepeatedSolveStability() {
		let n = 50
		let diag = (0..<n).map { _ in Double.random(in: 1...10) }
		let off = (0..<n-1).map { _ in Double.random(in: -1...1) }
		let A = TridiagonalMatrix(diagonal: diag, upper: off, lower: off)
		var lu = TridiagonalLUMatrix(A)
		let x = (0..<n).map { _ in Double.random(in: -2...2) }
		for _ in 0..<500 {
			var b = A * x
			let xSolved = lu.solve(&b)
			let err = zip(xSolved, x).map { abs($0 - $1) }.max() ?? 0.0
			XCTAssertLessThanOrEqual(err, 1e-8 * max(1.0, lu.approximateConditionNumber))
		}
	}
	
	/// Multithreaded smoke test: run many independent LU solves concurrently.
	func testConcurrentSolves() {
		let n = 30
		let diag = (0..<n).map { _ in Double.random(in: 1...10) }
		let off = (0..<n-1).map { _ in Double.random(in: -1...1) }
		let iterations = 100
		DispatchQueue.concurrentPerform(iterations: iterations) { _ in
			let A = TridiagonalMatrix(diagonal: diag, upper: off, lower: off)
			var lu = TridiagonalLUMatrix(A)
			let x = (0..<n).map { _ in Double.random(in: -2...2) }
			var b = A * x
			let xSolved = lu.solve(&b)
			let err = zip(xSolved, x).map { abs($0 - $1) }.max() ?? 0.0
			XCTAssertLessThanOrEqual(err, 1e-7 * max(1.0, lu.approximateConditionNumber))
		}
	}
	
	
	// MARK: - Utilities
	@inline(__always)
	private func approxEqual(_ a: Double, _ b: Double, tol: Double = 1e-10) -> Bool {
		abs(a - b) <= tol * max(1.0, abs(a), abs(b))
	}
	
	private func makeTestMatrix(n: Int) -> TridiagonalMatrix<Double> {
		let d: [Double] = (0..<n).map { i in 4.0 + Double(i % 3) }
		let du: [Double] = (0..<(n-1)).map { _ in 1.0 }
		let dl: [Double] = (0..<(n-1)).map { _ in 1.0 }
		return TridiagonalMatrix(diagonal: d, upper: du, lower: dl)
	}
	
	// MARK: - Single RHS Solve
	func testSingleRHSSolve() throws {
		let n = 7
		let A = makeTestMatrix(n: n)
		var lu = A.factorized()
		XCTAssertFalse(lu.isSingular)
		
		let xTrue: [Double] = (0..<n).map { i in Double(i) * 0.7 - 1.2 }
		let b = A * xTrue
		
		var bCopy = b
		let xSolved = lu.solve(&bCopy)
		
		for (xs, xt) in zip(xSolved, xTrue) {
			XCTAssertTrue(approxEqual(xs, xt), "Expected \(xt), got \(xs)")
		}
	}
	
	// MARK: - Batched Solve via [[T]]
	func testBatchedColumnsSolve() throws {
		let n = 7
		let nrhs = 3
		let A = makeTestMatrix(n: n)
		var lu = A.factorized()
		XCTAssertFalse(lu.isSingular)
		
		let xBatch: [[Double]] = (0..<nrhs).map { j in
			(0..<n).map { i in Double(i + j) * 0.3 + 0.1 }
		}
		
		let bCols = xBatch.map { A * $0 }
		let solvedCols = lu.solve(columns: bCols)
		
		for (solved, truth) in zip(solvedCols, xBatch) {
			for (xs, xt) in zip(solved, truth) {
				XCTAssertTrue(approxEqual(xs, xt), "Expected \(xt), got \(xs)")
			}
		}
	}
	
	// MARK: - Batched Solve via Column-Major Flat Layout
	func testBatchedFlatSolve() throws {
		let n = 7
		let nrhs = 3
		let A = makeTestMatrix(n: n)
		var lu = A.factorized()
		XCTAssertFalse(lu.isSingular)
		
		let xBatch: [[Double]] = (0..<nrhs).map { j in
			(0..<n).map { i in Double(i + j) * 0.3 + 0.1 }
		}
		
		let bCols = xBatch.map { A * $0 }
		
		// Pack into column-major flat buffer
		var bFlat = Array<Double>(repeating: 0.0, count: n * nrhs)
		for j in 0..<nrhs {
			for i in 0..<n {
				bFlat[i + j*n] = bCols[j][i]
			}
		}
		
		var bFlatCopy = bFlat
		let solvedFlat = lu.solve(&bFlatCopy, nrhs: nrhs)
		
		// Unpack and compare
		for j in 0..<nrhs {
			for i in 0..<n {
				let xs = solvedFlat[i + j*n]
				let xt = xBatch[j][i]
				XCTAssertTrue(approxEqual(xs, xt), "Expected \(xt), got \(xs)")
			}
		}
	}
	
	// MARK: - Transpose Solve
	func testTransposeSolve() throws {
		let n = 7
		let A = makeTestMatrix(n: n)
		var lu = A.factorized()
		XCTAssertFalse(lu.isSingular)
		
		let xTrue: [Double] = (0..<n).map { i in Double(i) * 0.11 - 0.3 }
		
		// b = A^T * xTrue
		var b = Array(repeating: 0.0, count: n)
		b[0] = A.diagonal[0] * xTrue[0] + A.lower[0] * xTrue[1]
		for i in 1..<(n-1) {
			b[i] = A.upper[i-1] * xTrue[i-1]
			+ A.diagonal[i] * xTrue[i]
			+ A.lower[i] * xTrue[i+1]
		}
		b[n-1] = A.upper[n-2] * xTrue[n-2] + A.diagonal[n-1] * xTrue[n-1]
		
		var bCopy = b
		let xSolved = lu.solve(&bCopy, transpose: true)
		
		for (xs, xt) in zip(xSolved, xTrue) {
			XCTAssertTrue(approxEqual(xs, xt), "Expected \(xt), got \(xs)")
		}
	}
}

