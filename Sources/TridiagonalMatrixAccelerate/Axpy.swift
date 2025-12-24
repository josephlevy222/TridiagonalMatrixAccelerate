//
//  Axpy.swift
//  TridiagonalMatrixAccelerate
//
//  Created by Joseph Levy on 12/3/25.
//
import Accelerate
import Numerics

@inlinable public func AXpY_<T: RealScalar>( _ A: TridiagonalMatrix<T>, _ x: ColumnVector<T>, _ y: inout ColumnVector<T>
) -> ColumnVector<T> {
	let n = x.count
	precondition(n == A.size)
	
	if n == 0 { return [] }
	
	if n == 1 {
		y[0] = A.diagonal[0] * x[0]
		return y
	}
	x.withUnsafeBufferPointer { xPtr in
		y.withUnsafeMutableBufferPointer { yPtr in
			// y = d * x
			A.diagonal.withUnsafeBufferPointer { dPtr in
				T.vma(dPtr.baseAddress!, 1, xPtr.baseAddress!, 1, yPtr.baseAddress!, 1, n )
			}
			// y[0...n-2] += upper * x[1...n-1]
			A.upper.withUnsafeBufferPointer { uPtr in
				T.vma(uPtr.baseAddress!, 1, xPtr.baseAddress! + 1, 1, yPtr.baseAddress!, 1, n - 1 )
			}
			// y[1...n-1] += lower * x[0...n-2]
			A.lower.withUnsafeBufferPointer { lPtr in
				T.vma( lPtr.baseAddress!, 1, xPtr.baseAddress!, 1, yPtr.baseAddress! + 1, 1, n - 1 )
			}
			
		}
	}
	return y
}

/// Helper: Complex multiply-add for a band: y += band * x
@inline(__always) public func complexBandMA<T: RealScalar>(_ band: [Complex<T>],_ x: UnsafePointer<T>,_ y: CMutablePtr<T>,
														   _ temp: CMutablePtr<T>,_ count: Int ) {
	band.withUnsafeBufferPointer { bandPtr in
		let bandBase = bandPtr.baseAddress!
		bandBase.withMemoryRebound(to: T.self, capacity: 2*count) { b in
			T.vma(b, 2, x, 2, y, 2, count)          // y.real += d.real * x.real
			T.vmul(b+1, 2, x+1, 2, temp, 1, count)  // y.real += d.real * x.imag
			T.vsub(temp, 1, y, 2, y, 2, count)      // y.real = y.real - temp
			T.vma(b, 2, x+1, 2, y+1, 2, count)      // y.imag += d.real * x.imag
			T.vma(b+1, 2, x, 2, y+1, 2, count)      // y.imag += d.imag * x.real
		}
	}
}

@inlinable public func AXpY_<T: RealScalar>(
	_ A: TridiagonalMatrix<Complex<T>>, _ x: [Complex<T>], _ y: inout [Complex<T>]
) -> [Complex<T>] {
	
	let n = x.count
	precondition(n == A.size)
	precondition(y.count == n)
	
	if n == 0 { return y }
	
	if n == 1 {
		y[0] = y[0] + A.diagonal[0] * x[0]
		return y
	}
	
	// We need temp of size of diagonal = n, but off-diagonals only use n-1
	let tempSize = n  // For diagonal computations
	
	if tempSize <= 1024 {
		// Small: Use stack for both
		withUnsafeTemporaryAllocation(of: T.self, capacity: tempSize) { diagTemp in
			computeAXpYWithTemps(A, x: x, y: &y, diagTemp: diagTemp.baseAddress!)
		}
	} else {
		// Large: Use heap
		let diagTemp = UnsafeMutableBufferPointer<T>.allocate(capacity: tempSize)
		defer { diagTemp.deallocate() }
		computeAXpYWithTemps(A, x: x, y: &y, diagTemp: diagTemp.baseAddress!)
	}
	return y
}

// Version that reuses same temp array for both diagonal and off-diagonal
@inlinable internal func computeAXpYWithTemps<T: RealScalar>(
	_ A: TridiagonalMatrix<Complex<T>>, x: [Complex<T>], y: inout [Complex<T>],  diagTemp: UnsafeMutablePointer<T>
) {
	let n = x.count
	
	x.withUnsafeBufferPointer { xPtr in
		y.withUnsafeMutableBufferPointer { yPtr in
			
			let xBase = xPtr.baseAddress!
			let yBase = yPtr.baseAddress!
			
			xBase.withMemoryRebound(to: T.self, capacity: 2*n) { x in
				yBase.withMemoryRebound(to: T.self, capacity: 2*n) { y in
					// --- DIAGONAL ---
					complexBandMA(A.diagonal, x, y, diagTemp, n)
					// --- UPPER DIAGONAL ---
					complexBandMA(A.upper, x+2, y, diagTemp, n-1)
					// --- LOWER DIAGONAL ---
					complexBandMA(A.lower, x, y+2, diagTemp, n-1)
				}
			}
			
		}
	}
}

// Real scalar optimized implementation
@inlinable public func multiply_<T: RealScalar>(_ A: TridiagonalMatrix<T>, _ x: ColumnVector<T>) -> ColumnVector<T> {
	precondition(x.count == A.size, "Invalid column vector size")
	let n = x.count
	if n == 0 { return [] }
	if n == 1 { return [A.diagonal[0] * x[0]] }
	
	var result = [T](unsafeUninitializedCapacity: n) { buffer, initializedCount in
		initializedCount = n
	}
	
	x.withUnsafeBufferPointer { xPtr in
		result.withUnsafeMutableBufferPointer { resultPtr in
			let xBase = xPtr.baseAddress!
			let r = resultPtr.baseAddress!
			
			A.diagonal.withUnsafeBufferPointer { dPtr in
				T.vmul(dPtr.baseAddress!, 1, xBase, 1, r, 1, n)
			}
			A.upper.withUnsafeBufferPointer { uPtr in
				T.vma(uPtr.baseAddress!, 1, xBase + 1, 1, r, 1, n-1)
			}
			A.lower.withUnsafeBufferPointer { lPtr in
				T.vma(lPtr.baseAddress!, 1, xBase, 1, r + 1, 1, n-1)
			}
		}
	}
	
	return result
}

// Complex scalar optimized implementation
@inlinable public func multiply_<T: RealScalar>(_ A: TridiagonalMatrix<Complex<T>>, _ x: ColumnVector<Complex<T>>
) -> ColumnVector<Complex<T>> {
	precondition(x.count == A.size, "Invalid column vector size")
	let n = x.count
	if n == 0 { return [] }
	if n == 1 { return [A.diagonal[0] * x[0]] }
	
	var result = [Complex<T>](unsafeUninitializedCapacity: n) { buffer, initializedCount in
		initializedCount = n
	}
	
	var temp = [T](unsafeUninitializedCapacity: n) { buffer, initializedCount in
		initializedCount = n
	}
	
	x.withUnsafeBufferPointer { xPtr in
		result.withUnsafeMutableBufferPointer { resultPtr in
			let xBase = xPtr.baseAddress!
			let resultBase = resultPtr.baseAddress!
			
			xBase.withMemoryRebound(to: T.self, capacity: 2*n) { x in
				resultBase.withMemoryRebound(to: T.self, capacity: 2*n) { y in
					A.diagonal.withUnsafeBufferPointer { dPtr in
						let dBase = dPtr.baseAddress!
						dBase.withMemoryRebound(to: T.self, capacity: 2*n) { d in
							T.vmul(d, 1, x, 1, y, 1, 2*n)
							T.vsub(y+1, 2, y, 2, y, 2, n)
							T.vmul(d, 2, x+1, 2, y+1, 2, n)
							T.vma(d+1, 2, x, 2, y+1, 2, n)
						}
					}
					complexBandMA(A.upper, x+2, y, &temp, n-1)
					complexBandMA(A.lower, x, y+2, &temp, n-1)
				}
			}
		}
	}
	return result
}

// MARK: -  Tridiagonal Matrix-Vector Multiplication Operator using protocol dispatch
@inlinable
public func *<T: ScalarField>(_ A: TridiagonalMatrix<T>, _ x: ColumnVector<T>) -> ColumnVector<T> { T.multiply(A, x) }

