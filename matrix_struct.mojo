import matmul
from sys.info import simdwidthof, is_apple_silicon
from memory import memcpy, memcmp, memset_zero, UnsafePointer
from algorithm import vectorize, parallelize
from buffer import NDBuffer
import algorithm
from collections import Set
import math
import random
from python import Python, PythonObject

struct Matrix(Stringable, Writable, Copyable, Movable, Sized):
    var height: Int
    var width: Int
    var size: Int
    var data: UnsafePointer[Float32]
    var order: String
    alias simd_width: Int = 4 * simdwidthof[DType.float32]() if is_apple_silicon() else 2 * simdwidthof[DType.float32]()

    # initialize from UnsafePointer
    @always_inline
    fn __init__(out self, data: UnsafePointer[Float32], height: Int, width: Int, order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = data
        self.order = order.lower()

    # initialize by copying from UnsafePointer
    @always_inline
    fn __init__(out self, height: Int, width: Int, data: UnsafePointer[Float32] = UnsafePointer[Float32](), order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = UnsafePointer[Float32].alloc(self.size)
        self.order = order.lower()
        if data:
            memcpy(self.data, data, self.size)

    # initialize from 2D List
    fn __init__(out self, def_input: List[List[Float32]]) raises:
        self.height = len(def_input)
        self.width = len(def_input[0]) if self.height > 0 else 0
        self.size = self.height * self.width
        self.data = UnsafePointer[Float32].alloc(self.size)
        self.order = 'c'
        if self.size > 0:
            for row_i in range(len(def_input)):
                memcpy(self.data + row_i * self.width, def_input[row_i].data, self.width)

    fn __copyinit__(out self, other: Self):
        self.height = other.height
        self.width = other.width
        self.size = other.size
        self.data = UnsafePointer[Float32].alloc(self.size)
        self.order = other.order
        memcpy(self.data, other.data, self.size)

    fn __moveinit__(out self, owned existing: Self):
        self.height = existing.height
        self.width = existing.width
        self.size = existing.size
        self.data = existing.data
        self.order = existing.order
        existing.height = existing.width = existing.size = 0
        existing.order = ''
        existing.data = UnsafePointer[Float32]()

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        var loc: Int
        if self.order == 'c':
            loc = (y * self.width) + x
        else:
            loc = (x * self.height) + y
        return self.data.load[width=nelts](loc)

    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        var loc: Int
        if self.order == 'c':
            loc = (y * self.width) + x
        else:
            loc = (x * self.height) + y
        return self.data.store(loc, val)

    # access an element
    @always_inline
    fn __getitem__(self, row: Int, column: Int) raises -> Float32:
        var loc: Int
        if self.order == 'c':
            loc = (row * self.width) + column
        else:
            loc = (column * self.height) + row
        if loc > self.size - 1:
            raise Error("Error: Location is out of range!")
        return self.data[loc]

    # access a row
    @always_inline
    fn __getitem__(self, row: Int) raises -> Matrix:
        if row >= self.height or row < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            return Matrix(1, self.width, self.data + (row * self.width), self.order)
        var mat = Matrix(1, self.width, order= self.order)
        var tmpPtr = self.data + row
        @parameter
        fn convert[simd_width: Int](idx: Int):
            mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.height))
            tmpPtr += simd_width * self.height
        vectorize[convert, self.simd_width](mat.width)
        return mat^

    # access a row (unsafe)
    @always_inline
    fn __getitem__(self, row: Int, *, unsafe: Bool) -> Matrix:
        if self.order == 'c' or self.height == 1:
            return Matrix(1, self.width, self.data + (row * self.width), self.order)
        var mat = Matrix(1, self.width, order= self.order)
        var tmpPtr = self.data + row
        @parameter
        fn convert[simd_width: Int](idx: Int):
            mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.height))
            tmpPtr += simd_width * self.height
        vectorize[convert, self.simd_width](mat.width)
        return mat^

    # access a row with offset
    @always_inline
    fn __getitem__(self, row: Int, offset: Bool, start_i: Int) raises -> Matrix:
        if row >= self.height or row < 0 or start_i >= self.width or start_i < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            return Matrix(1, self.width - start_i, self.data + (row * self.width) + start_i, self.order)
        var mat = Matrix(1, self.width - start_i, order= self.order)
        var tmpPtr = self.data + row + (start_i * self.height)
        @parameter
        fn convert[simd_width: Int](idx: Int):
            mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.height))
            tmpPtr += simd_width * self.height
        vectorize[convert, self.simd_width](mat.width)
        return mat^

    # access a column
    @always_inline
    fn __getitem__(self, row: String, column: Int) raises -> Matrix:
        if column >= self.width or column < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' and self.width > 1:
            var mat = Matrix(self.height, 1)
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.width))
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](mat.height)
            return mat^
        return Matrix(self.height, 1, self.data + (column * self.height), self.order)

    # access a column (unsafe)
    @always_inline
    fn __getitem__(self, row: String, column: Int, *, unsafe: Bool) -> Matrix:
        if self.order == 'c' and self.width > 1:
            var mat = Matrix(self.height, 1)
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.width))
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](mat.height)
            return mat^
        return Matrix(self.height, 1, self.data + (column * self.height), self.order)

    # access a column with offset
    @always_inline
    fn __getitem__(self, offset: Bool, start_i: Int, column: Int) raises -> Matrix:
        if column >= self.width or column < 0 or start_i >= self.height or start_i < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' and self.width > 1:
            var mat = Matrix(self.height - start_i, 1)
            var tmpPtr = self.data + column + (start_i * self.width)
            @parameter
            fn convert[simd_width: Int](idx: Int):
                mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.width))
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](mat.height)
            return mat^
        return Matrix(self.height - start_i, 1, self.data + (column * self.height) + start_i, self.order)

    # access given rows (by their indices)
    @always_inline
    fn __getitem__(self, rows: Matrix) raises -> Matrix:
        var mat = Matrix(rows.size, self.width, order= self.order)
        if rows.size > 96:
            @parameter
            fn p(i: Int):
                mat[i, unsafe=True] = self[Int(rows.data[i]), unsafe=True]
            parallelize[p](rows.size)
        else:
            for i in range(rows.size):
                mat[i] = self[Int(rows.data[i])]
        return mat^

    # access given columns (by their indices)
    @always_inline
    fn __getitem__(self, row: String, columns: Matrix) raises -> Matrix:
        var mat = Matrix(self.height, columns.size, order= self.order)
        if columns.size > 96 or (self.order == 'c' and self.height * columns.size > 24576):
            @parameter
            fn p(i: Int):
                mat[row, i, unsafe=True] = self[row, Int(columns.data[i]), unsafe=True]
            parallelize[p](columns.size)
        else:
            for i in range(columns.size):
                mat[row, i] = self[row, Int(columns.data[i])]
        return mat^

    # access given rows (by their indices)
    @always_inline
    fn __getitem__(self, rows: List[Int]) raises -> Matrix:
        var mat = Matrix(len(rows), self.width, order= self.order)
        if len(rows) > 96:
            @parameter
            fn p(i: Int):
                mat[i, unsafe=True] = self[rows[i], unsafe=True]
            parallelize[p](len(rows))
        else:
            for i in range(mat.height):
                mat[i] = self[rows[i]]
        return mat^

    # access given rows (by their indices)
    @always_inline
    fn __getitem__(self, rows: List[Scalar[DType.index]]) raises -> Matrix:
        var mat = Matrix(len(rows), self.width, order= self.order)
        if len(rows) > 96:
            @parameter
            fn p(i: Int):
                mat[i, unsafe=True] = self[rows[i].value, unsafe=True]
            parallelize[p](len(rows))
        else:
            for i in range(mat.height):
                mat[i] = self[rows[i].value]
        return mat^

    # access given columns (by their indices)
    @always_inline
    fn __getitem__(self, row: String, columns: List[Int]) raises -> Matrix:
        var mat = Matrix(self.height, len(columns), order= self.order)
        if len(columns) > 96 or (self.order == 'c' and self.height * len(columns) > 24576):
            @parameter
            fn p(i: Int):
                mat[row, i, unsafe=True] = self[row, columns[i], unsafe=True]
            parallelize[p](len(columns))
        else:
            for i in range(mat.width):
                mat[row, i] = self[row, columns[i]]
        return mat^

    # access given columns (by their indices)
    @always_inline
    fn __getitem__(self, row: String, columns: List[Scalar[DType.index]]) raises -> Matrix:
        var mat = Matrix(self.height, len(columns), order= self.order)
        if len(columns) > 96 or (self.order == 'c' and self.height * len(columns) > 24576):
            @parameter
            fn p(i: Int):
                mat[row, i, unsafe=True] = self[row, columns[i].value, unsafe=True]
            parallelize[p](len(columns))
        else:
            for i in range(mat.width):
                mat[row, i] = self[row, columns[i].value]
        return mat^
    
    # replace an element
    @always_inline
    fn __setitem__(mut self, row: Int, column: Int, val: Float32) raises:
        var loc: Int
        if self.order == 'c':
            loc = (row * self.width) + column
        else:
            loc = (column * self.height) + row
        if loc > self.size - 1:
            raise Error("Error: Location is out of range!")
        self.data[loc] = val
    
    # replace the given row
    @always_inline
    fn __setitem__(mut self, row: Int, val: Matrix) raises:
        if row >= self.height or row < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            memcpy(self.data + (row * self.width), val.data, val.size)
        else:
            var tmpPtr = self.data + row
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.height)
                tmpPtr += simd_width * self.height
            vectorize[convert, self.simd_width](val.size)

    # replace the given row (unsafe)
    @always_inline
    fn __setitem__(mut self, row: Int, val: Matrix, *, unsafe: Bool):
        if self.order == 'c' or self.height == 1:
            memcpy(self.data + (row * self.width), val.data, val.size)
        else:
            var tmpPtr = self.data + row
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.height)
                tmpPtr += simd_width * self.height
            vectorize[convert, self.simd_width](val.size)

    # replace the given row with offset
    @always_inline
    fn __setitem__(mut self, row: Int, offset: Bool, start_i: Int, val: Matrix) raises:
        if row >= self.height or row < 0 or start_i >= self.width or start_i < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            memcpy(self.data + (row * self.width) + start_i, val.data, val.size)
        else:
            var tmpPtr = self.data + row + (start_i * self.height)
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.height)
                tmpPtr += simd_width * self.height
            vectorize[convert, self.simd_width](val.size)

    # replace the given column
    @always_inline
    fn __setitem__(mut self, row: String, column: Int, val: Matrix) raises:
        if column >= self.width or column < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' and self.width > 1:
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.width)
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](val.size)
        else:
            memcpy(self.data + (column * self.height), val.data, val.size)

    # replace the given column (unsafe)
    @always_inline
    fn __setitem__(mut self, row: String, column: Int, val: Matrix, *, unsafe: Bool):
        if self.order == 'c' and self.width > 1:
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.width)
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](val.size)
        else:
            memcpy(self.data + (column * self.height), val.data, val.size)

    # replace the given column with offset
    @always_inline
    fn __setitem__(mut self, offset: Bool, start_i: Int, column: Int, val: Matrix) raises:
        if column >= self.width or column < 0 or start_i >= self.height or start_i < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' and self.width > 1:
            var tmpPtr = self.data + column + (start_i * self.width)
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.width)
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](val.size)
        else:
            memcpy(self.data + (column * self.height) + start_i, val.data, val.size)

    # replace given rows (by their indices)
    @always_inline
    fn __setitem__(mut self, rows: Matrix, rhs: Matrix) raises:
        for i in range(rows.size):
            self[Int(rows.data[i])] = rhs[i]

    # replace given columns (by their indices)
    @always_inline
    fn __setitem__(mut self, row: String, columns: Matrix, rhs: Matrix) raises:
        for i in range(columns.size):
            self[row, Int(columns.data[i])] = rhs[row, i]
    
    @always_inline
    fn load_columns(self, _range: Int) raises -> Matrix:
        if _range > self.width:
            raise Error("Error: Index out of range!")
        var mat = Matrix(self.height, _range, order=self.order)
        if self.order == 'f' or self.height == 1:
            memcpy(mat.data, self.data, mat.size)
        else:
            @parameter
            fn p(i: Int):
                memcpy(mat.data + i * _range, self.data + i * self.width, _range)
            parallelize[p](self.height)
        return mat^

    @always_inline
    fn load_rows(self, _range: Int) raises -> Matrix:
        if _range > self.height:
            raise Error("Error: Index out of range!")
        var mat = Matrix(_range, self.width, order=self.order)
        if self.order == 'c' or self.width == 1:
            memcpy(mat.data, self.data, mat.size)
        else:
            @parameter
            fn p(i: Int):
                memcpy(mat.data + i * _range, self.data + i * self.height, _range)
            parallelize[p](self.width)
        return mat^

    @always_inline
    fn __del__(owned self):
        if self.data:
            self.data.free()

    @always_inline
    fn __len__(self) -> Int:
        return self.size

    @always_inline
    fn __eq__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] == rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] == rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __ne__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] != rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] != rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __gt__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] > rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] > rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __ge__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] >= rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] >= rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __lt__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] < rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] < rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __le__(self, rhs: Float32) -> List[Bool]:
        var result = List[Bool](capacity=self.size)
        result.resize(self.size, False)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] <= rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] <= rhs
            parallelize[cmp](self.size)
        return result^

    @always_inline
    fn __eq__(self, rhs: Self) -> Bool:
        return self.height == rhs.height and self.width == rhs.width and memcmp(self.data, rhs.data, self.size) == 0

    @always_inline
    fn __ne__(self, rhs: Self) -> Bool:
        return not self == rhs

    @always_inline
    fn __mul__(self, rhs: Self) raises -> Self:
        if self.width != rhs.height:
            raise Error('Error: Cannot multiply matrices with shapes (' + String(self.height) + ', ' + String(self.width) + ') and (' + String(rhs.height) + ', ' + String(rhs.width) + ')')

        var A = matmul.Matrix[DType.float32](self.data, (self.height, self.width))
        var B = matmul.Matrix[DType.float32](rhs.data, (rhs.height, rhs.width))
        var C = matmul.Matrix[DType.float32]((self.height, rhs.width))
        memset_zero(C.data, self.height * rhs.width)
        matmul.matmul(self.height, self.width, rhs.width, C, A, B)
        return Matrix(C.data, self.height, rhs.width)

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        var res: String = "["
        var strings = List[String]()
        for i in range(self.width):
            var max_len: Int = 0
            for j in range(self.height):
                strings.append("")
                var val = self.load[1](j, i)
                if val >= 0:
                    strings[j] += " "
                strings[j] += String(val)
                if len(strings[j]) > max_len:
                    max_len = len(strings[j])
            for j in range(self.height):
                var rng: Int = max_len - len(strings[j]) + 1
                for _ in range(rng):
                    strings[j] += " "

        for i in range(self.height):
            if i != 0:
                res += " "
            res += "[" + strings[i] + "]"
            if i != self.height - 1:
                res += "\n"
        writer.write(res + "]")