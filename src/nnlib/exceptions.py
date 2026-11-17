"""Hệ thống các lớp ngoại lệ chuẩn cho thư viện mạng nơ-ron nnlib."""


class NNError(Exception):
    """Lớp cơ sở cho toàn bộ ngoại lệ trong thư viện mạng nơ-ron nnlib.

    Mọi ngoại lệ do tầng lõi sinh ra đều kế thừa từ lớp này để tầng Controller
    hoặc ứng dụng có thể bắt và xử lý một cách nhất quán.
    """

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self._message: str = message

    @property
    def message(self) -> str:
        """Thông điệp chi tiết của ngoại lệ."""
        return self._message

    def __str__(self) -> str:
        return self._message or self.__class__.__name__


class ShapeMismatchError(NNError):
    """Ngoại lệ khi kích thước (shape) của tensor hoặc ma trận không khớp với kỳ vọng.

    Thường được kích hoạt khi kiểm tra shape đầu vào của Layer (ví dụ Rule R1).
    """


class InvalidConfigError(NNError):
    """Ngoại lệ khi cấu hình mô hình, siêu tham số hoặc tham số khởi tạo không hợp lệ.

    Ví dụ: learning rate <= 0, dropout rate không thuộc [0, 1), out_features không
    khớp in_features (Rule R3, R5, R7).
    """


class StateError(NNError):
    """Ngoại lệ khi trạng thái thực thi của mô hình hoặc layer không hợp lệ.

    Ví dụ: gọi backward trước khi forward hoặc khi không tồn tại cache (Rule R2).
    """


class ModelFileError(NNError):
    """Ngoại lệ khi có lỗi trong quá trình lưu hoặc nạp file mô hình.

    Ví dụ: file không tồn tại, định dạng hỏng, sai version hoặc shape tham số
    không khớp với cấu hình (Rule R6).
    """
