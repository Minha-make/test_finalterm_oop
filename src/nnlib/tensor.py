"""Định nghĩa lớp Tensor và Parameter đại diện cho dữ liệu và trọng số mô hình."""

from __future__ import annotations

import numpy as np


class Tensor:
    """Lớp bọc mảng nhiều chiều ndarray với cơ chế kiểm tra shape.

    Quy ước chung: Dữ liệu tính toán luôn ở dạng batch-first: (N, features).
    Kiểu dữ liệu mặc định là float32 khi huấn luyện, float64 khi gradient check.
    """

    def __init__(
        self,
        data: np.ndarray | list | float | int,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        """Khởi tạo Tensor từ mảng số hoặc dữ liệu có thể chuyển đổi thành ndarray.

        Args:
            data: Dữ liệu mảng, danh sách hoặc số vô hướng.
            dtype: Kiểu dữ liệu số học (mặc định np.float32).
        """
        if isinstance(data, np.ndarray):
            if data.dtype != dtype:
                self._data: np.ndarray = data.astype(dtype)
            else:
                self._data = data
        else:
            self._data = np.array(data, dtype=dtype)

    @property
    def data(self) -> np.ndarray:
        """Trả về mảng ndarray bên dưới."""
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """Gán mảng dữ liệu mới cho tensor."""
        if not isinstance(value, np.ndarray):
            raise TypeError("Dữ liệu gán vào tensor phải là numpy.ndarray.")
        self._data = value

    @property
    def shape(self) -> tuple[int, ...]:
        """Trả về kích thước các chiều của tensor."""
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        """Trả về kiểu dữ liệu phần tử của tensor."""
        return self._data.dtype

    @property
    def ndim(self) -> int:
        """Trả về số chiều của tensor."""
        return self._data.ndim

    def __array__(self) -> np.ndarray:
        """Hỗ trợ chuyển đổi tự nhiên sang numpy ndarray."""
        return self._data

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"


class Parameter(Tensor):
    """Trọng số có thể học được của mô hình, kế thừa từ Tensor và tích lũy gradient."""

    def __init__(
        self,
        data: np.ndarray | list | float | int,
        name: str = "",
        dtype: np.dtype | type = np.float32,
    ) -> None:
        """Khởi tạo Parameter.

        Args:
            data: Trọng số khởi tạo.
            name: Tên của tham số phục vụ việc lưu trữ và nạp trọng số.
            dtype: Kiểu dữ liệu của tham số (mặc định np.float32).
        """
        super().__init__(data, dtype=dtype)
        self._name: str = name
        self._grad: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Tên định danh của tham số."""
        return self._name

    @property
    def grad(self) -> np.ndarray | None:
        """Đạo hàm riêng tích lũy, có cùng shape với data."""
        return self._grad

    @grad.setter
    def grad(self, value: np.ndarray | None) -> None:
        """Cập nhật gradient tích lũy."""
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("Gradient phải là numpy.ndarray hoặc None.")
        if value is not None and value.shape != self.shape:
            raise ValueError(
                f"Gradient shape {value.shape} không khớp "
                f"với parameter shape {self.shape}."
            )
        self._grad = value

    def zero_grad(self) -> None:
        """Xóa trắng gradient tích lũy về ma trận 0 có cùng kích thước."""
        self._grad = np.zeros_like(self._data)

    def __repr__(self) -> str:
        return f"Parameter(name='{self._name}', shape={self.shape}, dtype={self.dtype})"
