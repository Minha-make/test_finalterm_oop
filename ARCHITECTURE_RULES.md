# ARCHITECTURE RULES & CODE CONVENTIONS
## Dự án: Thư viện mạng nơ-ron OOP & Ứng dụng nhận diện chữ số viết tay (MNIST)

Tài liệu này là kim chỉ nam kiến trúc, thiết kế hướng đối tượng (OOP) và quy chuẩn mã nguồn bắt buộc cho toàn bộ các phiên làm việc và các thành viên tham gia phát triển dự án `neuralnet-oop`. Mọi dòng mã được viết ra đều phải tuân thủ nghiêm ngặt các quy định trong tài liệu này.

---

## 1. Kiến trúc 4 tầng nghiêm ngặt (Strict 4-Tier Architecture)

Hệ thống được thiết kế theo mô hình 4 tầng phân tách độc lập (Separation of Concerns). Luồng phụ thuộc và lời gọi hàm **chỉ được phép đi từ trên xuống dưới (Strict Top-Down Dependency)**. Tuyệt đối không có chiều ngược lại.

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Tầng Giao diện (Presentation Layer)                      │
│    digitapp.cli (CliApp)                                    │
│    digitapp.gui (DrawingCanvasView, RecognitionPresenter)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (chỉ gọi controller)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Tầng Điều khiển / Ứng dụng (Controller Layer)            │
│    digitapp.controllers:                                    │
│    - ModelController (UC1 - UC4)                            │
│    - TrainController (UC5), EvaluateController (UC6)        │
│    - RecognitionController (UC7, UC8)                       │
└──────────────────────────────┬──────────────────────────────┘
                               │ (điều phối nghiệp vụ lõi)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Tầng Lõi / Nghiệp vụ & Thuật toán (Domain / Core Layer)  │
│    - nnlib (tensor, layers, initializers, losses, optim,    │
│             training, testing, exceptions)                  │
│    - digitapp core (data, metrics, predictor)               │
└──────────────────────────────┬──────────────────────────────┘
                               │ (lõi chỉ biết interface)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Tầng Hạ tầng & Cấu hình (Infrastructure & Config Layer)  │
│    - nnlib.io (NpzModelRepository cài đặt IModelRepository) │
│    - digitapp.config (AppConfig đọc config/config.json)     │
│    - digitapp.common.paths (resource_path)                  │
│    - Tệp dữ liệu MNIST IDX, thư mục models/, data/raw/      │
└─────────────────────────────────────────────────────────────┘
```

### Các quy tắc chiều phụ thuộc bắt buộc:
1. **Giao diện chỉ gọi Controller**: `digitapp.cli` và `digitapp.gui` chỉ được phép `import` và gọi các hàm của `digitapp.controllers`. Tuyệt đối không gọi trực tiếp các lớp nghiệp vụ trong `nnlib` hay `digitapp.data/predictor`.
2. **Controller không gọi ngược View**: Tầng Controller đóng vai trò điều phối tác vụ (Use Case), nhận dữ liệu từ View, gọi Tầng Lõi xử lý, bắt lỗi (`try/except`) và trả kết quả hoặc ném ngoại lệ có kiểm soát lên View. Controller không chứa tham chiếu tới widget của GUI hay in trực tiếp ra CLI.
3. **Lõi không biết Giao diện & Ứng dụng**: `nnlib` tuyệt đối không được phép `import digitapp`.
4. **Phân rã nội bộ trong nnlib**: `nnlib.layers` không được phép `import` `nnlib.optim` hay `nnlib.training`.
5. **Đảo ngược phụ thuộc (DIP)**: Tầng Lõi chỉ phụ thuộc vào các Interface trừu tượng (`IModelRepository`, `IDataset`), không phụ thuộc vào lớp cài đặt cụ thể ở tầng hạ tầng. Tầng hạ tầng sẽ cài đặt (implement) các interface đó.
6. **Kiểm tra tự động (Test I2)**: Dự án có test case tích hợp kiến trúc (`test_architecture.py`) sử dụng thư viện `ast` để quét toàn bộ câu lệnh `import`, tự động phát hiện và đánh trượt nếu có vi phạm import ngược tầng.

---

## 2. Chữ ký chính xác của 5 Hợp đồng giao tiếp (Contract Interfaces - v0.1-contract)

Hai phân hệ A (lõi `nnlib`) và B (huấn luyện & ứng dụng `digitapp`) liên kết với nhau thông qua 5 Interface/Lớp cốt lõi được chốt chữ ký. Mọi thay đổi đối với các chữ ký này đều phải có sự đồng thuận của cả 2 phân hệ.

### 2.1. Tensor & Parameter (`src/nnlib/tensor.py`)
```python
import numpy as np

class Tensor:
    """Lớp bọc mảng nhiều chiều ndarray với cơ chế kiểm tra shape.
    
    Quy ước: Dữ liệu luôn ở dạng batch-first: (N, features).
    """
    def __init__(self, data: np.ndarray | list | float | int) -> None:
        ...

    @property
    def data(self) -> np.ndarray:
        """Trả về mảng ndarray bên dưới."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Trả về kích thước các chiều của tensor."""
        ...

    @property
    def dtype(self) -> np.dtype:
        """Kiểu dữ liệu phần tử (float32 khi huấn luyện, float64 khi grad check)."""
        ...


class Parameter(Tensor):
    """Trọng số có thể học được của mô hình, kế thừa từ Tensor và tích lũy gradient."""
    def __init__(self, data: np.ndarray, name: str = "") -> None:
        ...

    @property
    def grad(self) -> np.ndarray | None:
        """Đạo hàm riêng tích lũy, cùng shape với data."""
        ...

    @property
    def name(self) -> str:
        """Tên của tham số phục vụ lưu trữ và nạp trọng số."""
        ...

    def zero_grad(self) -> None:
        """Xóa trắng gradient tích lũy về 0 (hoặc None)."""
        ...
```

### 2.2. Layer (`src/nnlib/layers/base.py`)
```python
from abc import ABC, abstractmethod
import numpy as np
from nnlib.tensor import Tensor, Parameter

class Layer(ABC):
    """Lớp cơ sở trừu tượng cho mọi tầng nơ-ron (Component trong Composite Pattern)."""

    def __init__(self, name: str = "") -> None:
        self._name: str = name
        self._training: bool = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def training(self) -> bool:
        return self._training

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Lan truyền thuận."""
        ...

    @abstractmethod
    def backward(self, grad_out: Tensor) -> Tensor:
        """Lan truyền ngược, nhận gradient từ tầng sau và trả về gradient đối với đầu vào."""
        ...

    def parameters(self) -> list[Parameter]:
        """Trả về danh sách tất cả các Parameter có thể học của tầng."""
        return []

    def train(self) -> None:
        """Chuyển sang chế độ huấn luyện (Dropout hoạt động)."""
        self._training = True

    def eval(self) -> None:
        """Chuyển sang chế độ suy luận / đánh giá (Dropout vô hiệu hóa)."""
        self._training = False

    def state_dict(self) -> dict[str, np.ndarray]:
        """Xuất từ điển trạng thái chứa trọng số của tầng."""
        ...

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Nạp trọng số từ từ điển trạng thái."""
        ...
```

### 2.3. ILossFunction (`src/nnlib/losses/base.py`)
```python
from abc import ABC, abstractmethod
from nnlib.tensor import Tensor

class ILossFunction(ABC):
    """Giao diện hàm mất mát (Strategy trong Strategy Pattern)."""

    @abstractmethod
    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Tính giá trị mất mát trung bình của batch."""
        ...

    @abstractmethod
    def backward(self) -> Tensor:
        """Tính đạo hàm dL / dy_pred.
        
        QUY ƯỚC QUAN TRỌNG: Gradient trả về PHẢI ĐƯỢC CHIA TRƯỚC CHO N (kích thước batch),
        để các Layer phía trước không cần chia lại N khi tính gradient tham số.
        """
        ...
```

### 2.4. IOptimizer (`src/nnlib/optim/base.py`)
```python
from abc import ABC, abstractmethod
from typing import Sequence
from nnlib.tensor import Parameter

class IOptimizer(ABC):
    """Giao diện thuật toán tối ưu hóa (Strategy trong Strategy Pattern)."""

    def __init__(self, lr: float) -> None:
        if lr <= 0:
            raise ValueError("Learning rate phải lớn hơn 0.")
        self._lr: float = lr

    @property
    def lr(self) -> float:
        return self._lr

    @abstractmethod
    def step(self, params: Sequence[Parameter]) -> None:
        """Cập nhật giá trị tham số dựa trên gradient tích lũy."""
        ...

    @abstractmethod
    def zero_grad(self, params: Sequence[Parameter]) -> None:
        """Xóa gradient tích lũy của toàn bộ tham số truyền vào."""
        ...
```

### 2.5. IModelRepository (`src/nnlib/models/repository.py`)
```python
from abc import ABC, abstractmethod
from nnlib.layers.base import Layer

class IModelRepository(ABC):
    """Giao diện lưu trữ và khôi phục mô hình (Repository Pattern / DIP)."""

    @abstractmethod
    def save(self, model: Layer, path: str) -> None:
        """Lưu kiến trúc (JSON) và trọng số (.npz) của mô hình xuống đĩa."""
        ...

    @abstractmethod
    def load(self, path: str) -> Layer:
        """Nạp cấu trúc và trọng số, tái tạo đối tượng mô hình hợp lệ."""
        ...
```

### Quy ước tính toán & kiểu dữ liệu chung:
- **Batch format**: Dạng `(N, features)`, trong đó `N` là kích thước mini-batch. Nhãn phân loại một-hot: `(N, 10)`.
- **Độ chính xác số thực**:
  - `np.float32`: Sử dụng mặc định trong quá trình huấn luyện và dự đoán để đảm bảo hiệu năng và tiết kiệm bộ nhớ.
  - `np.float64`: Chỉ sử dụng khi chạy kiểm tra gradient giải tích (`GradientChecker`) để triệt tiêu sai số làm tròn số học.
- **Ngoại lệ chuẩn**: Mọi ngoại lệ của thư viện kế thừa từ `NNError`.

---

## 3. Danh sách 12 Quy tắc nghiệp vụ (Business Rules: R1–R12)

Mỗi quy tắc nghiệp vụ dưới đây được gán mã từ **R1 đến R12**. Khi cài đặt trong mã nguồn, **BẮT BUỘC** phải gắn comment dạng:
`# Rule R<x>: <mô tả ngắn>` ngay tại dòng logic thực thi tương ứng.

| Mã Rule | Nội dung quy tắc | Nơi cài đặt | Phân hệ | Hành vi / Xử lý lỗi |
|:---|:---|:---|:---:|:---|
| **R1** | `Dense` chỉ nhận đầu vào ma trận 2 chiều `(N, in_features)`. | `Dense.forward` | A | Ném `ShapeMismatchError` nếu `x.ndim != 2` hoặc `x.shape[1] != self._in_features`. Bắt buộc comment `# Rule R1`. |
| **R2** | `backward` chỉ được gọi sau khi đã gọi `forward` (tồn tại cache dữ liệu). | `Layer.backward` (tất cả các lớp con) | A | Ném `StateError` nếu `self._cache_x is None` (hoặc biến cache tương ứng rỗng). Bắt buộc comment `# Rule R2`. |
| **R3** | `Dropout`: Xác suất giữ/tắt $0 \le p < 1$. Chỉ ngắt nơ-ron ở chế độ huấn luyện và nhân tỉ lệ bù $1/(1-p)$; ở chế độ suy luận (`eval`), giữ nguyên đầu vào. | `Dropout.__init__`, `Dropout.forward` | A | Ném `InvalidConfigError` nếu $p < 0$ hoặc $p \ge 1$. Kiểm tra `if self._training:`. Bắt buộc comment `# Rule R3`. |
| **R4** | Ổn định số học: `Softmax` trừ giá trị lớn nhất mỗi hàng trước khi tính $\exp(z_i - \max z)$; `CrossEntropyLoss` kẹp xác suất $p \in [\varepsilon, 1-\varepsilon]$ với $\varepsilon = 10^{-12}$ để tránh $\log(0)$. | `Softmax.forward`, `CrossEntropyLoss.forward` | A | Giữ giá trị hữu hạn, không sinh `NaN` hoặc `Inf`. Bắt buộc comment `# Rule R4`. |
| **R5** | Ràng buộc kích thước: Khi ghép nối mô hình, `out_features` của lớp `Dense` trước phải bằng chính xác `in_features` của lớp `Dense` sau. | `ModelBuilder.add`, `ModelBuilder.build` | A | Ném `InvalidConfigError` nếu chiều ra của lớp trước lệch chiều vào của lớp sau. Bắt buộc comment `# Rule R5`. |
| **R6** | Nạp mô hình: Phải kiểm tra phiên bản định dạng và tính khớp nhau giữa kích thước ma trận tham số trong file `.npz` với cấu hình `.json`. | `NpzModelRepository.load` | A | Ném `ModelFileError` hoặc `InvalidConfigError` nếu sai version hoặc kích thước trọng số không khớp kiến trúc. Bắt buộc comment `# Rule R6`. |
| **R7** | Hợp lệ hóa tối ưu: $\text{learning\_rate} > 0$, $\beta_1, \beta_2 \in [0, 1)$. Thuật toán `Adam` bắt buộc dùng bias correction cho bước thời gian $t$: $\hat{m} = \frac{m}{1-\beta_1^t}, \hat{v} = \frac{v}{1-\beta_2^t}$. | `IOptimizer.__init__`, `Adam.step` | B | Ném `InvalidConfigError` nếu siêu tham số ngoài khoảng hợp lệ. Bắt buộc comment `# Rule R7`. |
| **R8** | `EarlyStopping`: Theo dõi `val_loss`, nếu không giảm liên tiếp sau `patience` epoch thì dừng huấn luyện và khôi phục bộ trọng số tốt nhất (`best_weights`). | `EarlyStopping.on_epoch_end` | B | Gọi `trainer.request_stop()` và nạp lại trọng số tối ưu. Bắt buộc comment `# Rule R8`. |
| **R9** | Tách dữ liệu: Phân chia tập `train/val` theo tỉ lệ cố định với `seed` xác định; chỉ tập `train` được phép xáo trộn (shuffle) mỗi epoch, tập `val` giữ nguyên thứ tự. | `MnistDataset.split`, `DataLoader.__iter__` | B | Sử dụng Random State độc lập, không xáo trộn tập validation. Bắt buộc comment `# Rule R9`. |
| **R10** | Chuẩn hóa nét vẽ GUI: Cắt bounding box nhỏ nhất, thu phóng về kích thước $20 \times 20$ giữ nguyên tỉ lệ khung hình, định vị vào trung tâm ảnh $28 \times 28$ theo trọng tâm khối lượng (center of mass), chia 255 về $[0, 1]$. | `DigitPreprocessor.process` | B | Đảm bảo phân phối đặc trưng khớp với tập huấn luyện MNIST. Bắt buộc comment `# Rule R10`. |
| **R11** | Cảnh báo độ tin cậy: Nếu xác suất cao nhất của lớp dự đoán nhỏ hơn ngưỡng tin cậy `confidence_threshold` (mặc định 0.6 từ config) thì đánh dấu `is_confident = False` ("Không chắc chắn"). | `Predictor.predict` | B | Trả về đối tượng `Prediction` với thuộc tính `is_confident = False`. Bắt buộc comment `# Rule R11`. |
| **R12** | Canvas trống: Nếu tổng cường độ nét vẽ xấp xỉ 0 (người dùng chưa vẽ gì trên ô vẽ $280 \times 280$), không kích hoạt mô hình dự đoán. | `RecognitionController.recognize_drawing` | B | Ném hoặc xử lý `EmptyCanvasError`, hiển thị thông báo yêu cầu người dùng vẽ số. Bắt buộc comment `# Rule R12`. |

---

## 4. Danh sách 4 Design Patterns bắt buộc

Mỗi lớp tham gia vào Design Pattern **BẮT BUỘC** phải khai báo rõ vai trò của mình ngay trong **docstring** của lớp đó theo cú pháp:
`Vai trò trong Design Pattern: <Role> (<Pattern Name>)`.

### 4.1. Composite Pattern (Phân hệ A chủ trì)
- **Mục đích**: Xử lý một lớp đơn lẻ (`Dense`, `ReLU`) và một nhóm các lớp (`Sequential`) một cách đồng nhất thông qua interface `Layer`.
- **Các thành phần**:
  - `Component`: `Layer` (lớp trừu tượng định nghĩa hợp đồng `forward`, `backward`, `parameters`, `state_dict`).
  - `Leaf`: `Dense`, `Activation` (`ReLU`, `Sigmoid`, `Tanh`, `Softmax`), `Dropout`.
  - `Composite`: `Sequential` (chứa `_layers: list[Layer]`, duyệt `forward` xuôi, `backward` ngược, gom đệ quy `parameters()`).
- **Yêu cầu Docstring**:
  - `Layer`: `Vai trò trong Design Pattern: Component (Composite Pattern)`
  - `Dense` / `ReLU` / `Dropout`: `Vai trò trong Design Pattern: Leaf (Composite Pattern)`
  - `Sequential`: `Vai trò trong Design Pattern: Composite (Composite Pattern)`
- **Bài mở rộng**: Thêm lớp `Flatten` và kiểm thử `Sequential` lồng bên trong `Sequential` con mà không cần sửa code hiện có.

### 4.2. Factory Pattern với Registry (Phân hệ A chủ trì)
- **Mục đích**: Khởi tạo đối tượng `Layer` linh hoạt dựa trên chuỗi định danh cấu hình JSON mà không sử dụng chuỗi lệnh rẽ nhánh cứng `if/elif`.
- **Các thành phần**:
  - `Creator / Registry`: `LayerFactory` (chứa từ điển `_registry: dict[str, type[Layer]]`, decorator `@LayerFactory.register(name)` và hàm `LayerFactory.create(name, **kwargs)`).
  - `Product`: `Layer`.
  - `ConcreteProduct`: `Dense`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Dropout`.
- **Yêu cầu Docstring**:
  - `LayerFactory`: `Vai trò trong Design Pattern: Creator / Registry (Factory Pattern)`
  - Lớp cài đặt cụ thể: `Vai trò trong Design Pattern: ConcreteProduct (Factory Pattern)`
- **Bài mở rộng**: Thêm lớp `LeakyReLU`, đăng ký bằng decorator `@LayerFactory.register("leaky_relu")` và gọi được ngay từ file JSON cấu hình.

### 4.3. Strategy Pattern (Phân hệ B chủ trì)
- **Mục đích**: Tách rời thuật toán huấn luyện trong `Trainer` khỏi các thuật toán tối ưu cụ thể (`SGD`, `Adam`) và các hàm mất mát (`MSELoss`, `CrossEntropyLoss`), cho phép hoán đổi chiến lược trong thời gian chạy.
- **Các thành phần**:
  - `Context`: `Trainer`.
  - `Strategy`: `IOptimizer` và `ILossFunction`.
  - `ConcreteStrategy`: `SGD`, `Adam` (chiến lược tối ưu) và `MSELoss`, `CrossEntropyLoss` (chiến lược tính loss).
- **Yêu cầu Docstring**:
  - `Trainer`: `Vai trò trong Design Pattern: Context (Strategy Pattern)`
  - `IOptimizer` / `ILossFunction`: `Vai trò trong Design Pattern: Strategy Interface (Strategy Pattern)`
  - `SGD` / `Adam` / `CrossEntropyLoss`: `Vai trò trong Design Pattern: ConcreteStrategy (Strategy Pattern)`
- **Bài mở rộng**: Thêm thuật toán tối ưu `RMSprop` (`ConcreteStrategy`) mà không sửa đổi bất kỳ dòng mã nào trong `Trainer`.

### 4.4. Observer Pattern (Phân hệ B chủ trì)
- **Mục đích**: Tách các tác vụ giám sát, ghi log, lưu checkpoint, cập nhật thanh tiến trình giao diện và dừng sớm ra khỏi vòng lặp huấn luyện chính.
- **Các thành phần**:
  - `Subject`: `Trainer` (quản lý danh sách `_callbacks: list[ICallback]`).
  - `Observer`: `ICallback` (các phương thức sự kiện: `on_train_begin`, `on_epoch_begin`, `on_epoch_end`, `on_train_end`).
  - `ConcreteObserver`: `LoggingCallback`, `EarlyStopping`, `CheckpointCallback`, `ProgressCallback` (cho CLI/GUI).
- **Yêu cầu Docstring**:
  - `Trainer`: `Vai trò trong Design Pattern: Subject (Observer Pattern)`
  - `ICallback`: `Vai trò trong Design Pattern: Observer Interface (Observer Pattern)`
  - `EarlyStopping` / `LoggingCallback`: `Vai trò trong Design Pattern: ConcreteObserver (Observer Pattern)`
- **Bài mở rộng**: Thêm `CsvLoggerCallback` ghi lại nhật ký huấn luyện ra file CSV mà không ảnh hưởng tới `Trainer`.

---

## 5. Quy tắc đóng gói và Chuẩn mực lập trình (Encapsulation & Coding Standards)

### 5.1. Quy tắc đóng gói dữ liệu (Encapsulation)
1. **Thuộc tính ẩn danh**: Tất cả các thuộc tính bên trong lớp đều phải bắt đầu bằng một dấu gạch dưới `_ten_thuoctinh` (quy ước protected/private của Python).
   - Ví dụ: `self._data`, `self._grad`, `self._weights`, `self._bias`, `self._layers`, `self._lr`.
2. **Không truy cập trực tiếp**: Tuyệt đối không đọc hoặc gán giá trị trực tiếp cho thuộc tính `_ten_thuoctinh` từ bên ngoài lớp.
3. **Cung cấp `@property`**: Mọi truy cập thuộc tính từ bên ngoài đều phải thông qua `@property` (getter) và `@<ten>.setter` (chỉ khi thực sự cần cho phép sửa đổi và có kiểm tra tính hợp lệ).
4. **Kiểm tra tính hợp lệ tại `__init__`**: Mọi tham số đầu vào trong constructor phải được kiểm tra ràng buộc (như kích thước, khoảng giá trị).

### 5.2. Tuân thủ PEP 8
- Tên biến, hàm, phương thức, module, package: `snake_case` (ví dụ: `build_model`, `in_features`, `forward`).
- Tên lớp: `PascalCase` (ví dụ: `Dense`, `ModelController`, `NpzModelRepository`).
- Tên giao diện (Interface/ABC): Bắt đầu bằng chữ `I` và viết `PascalCase` (ví dụ: `ILossFunction`, `IOptimizer`, `IModelRepository`, `ICallback`, `IDataset`).
- Tên hằng số: `SCREAMING_SNAKE_CASE` (ví dụ: `DEFAULT_CONFIDENCE_THRESHOLD = 0.6`).
- Độ dài dòng: Tối đa 88 ký tự (chuẩn Black/Ruff).

### 5.3. Type Hints đầy đủ 100%
- Mọi hàm, phương thức, constructor đều phải có chú giải kiểu dữ liệu (type annotations) cho tất cả tham số và giá trị trả về.
- Hàm không trả về giá trị phải ghi rõ `-> None`.
- Sử dụng cú pháp hiện đại của Python 3.10+: `list[Layer]`, `tuple[int, ...]`, `dict[str, np.ndarray]`, `float | None`, `Sequence[Parameter]`.

### 5.4. Docstring tiếng Việt chuẩn
- 100% module, class và public method phải có docstring bằng tiếng Việt.
- **Class Docstring**: Mô tả mục đích của lớp, các thuộc tính chính và bắt buộc ghi rõ **Vai trò trong Design Pattern** nếu có.
- **Method Docstring**: Mô tả chức năng, các tham số (`Args:`), giá trị trả về (`Returns:`) và ngoại lệ phát sinh (`Raises:`).

### 5.5. Quản lý đường dẫn và tài nguyên
- **Không hardcode đường dẫn tuyệt đối**: Tuyệt đối cấm sử dụng chuỗi đường dẫn tuyệt đối dạng `C:\...` hay `/home/...`.
- **Hàm `resource_path`**: Mọi đường dẫn tới file cấu hình, trọng số mô hình, dữ liệu, icon đều phải đi qua hàm `resource_path(relative_path)` trong `src/digitapp/common/paths.py` để hỗ trợ chạy cả từ mã nguồn lẫn file đóng gói thực thi PyInstaller.

### 5.6. Xử lý ngoại lệ chuẩn hóa
- **Hệ thống lớp lỗi thống nhất**:
  - Gốc thư viện mạng nơ-ron: `NNError(Exception)`
    - `ShapeMismatchError`: Lỗi sai lệch kích thước tensor / ma trận (R1).
    - `InvalidConfigError`: Lỗi cấu hình mô hình hoặc siêu tham số không hợp lệ (R3, R5, R7).
    - `StateError`: Lỗi trạng thái thực thi không hợp lệ, ví dụ gọi backward trước forward (R2).
    - `ModelFileError`: Lỗi file mô hình bị hỏng hoặc không đúng định dạng (R6).
  - Ngoại lệ tầng ứng dụng: `AppError(NNError)`
    - `DataFormatError`: Lỗi định dạng tệp IDX / ảnh.
    - `EmptyCanvasError`: Lỗi khi người dùng bấm nhận diện trên canvas trống (R12).
- **Quy tắc bắt lỗi**:
  - Tầng lõi (`nnlib`): Chỉ phát hiện và `raise` ngoại lệ, không được nuốt lỗi.
  - Tuyệt đối cấm dùng `except:` trần (bare except) hoặc `except Exception: pass`.
  - Tầng Controller: `try/except` các loại lỗi cụ thể để ghi log và chuyển đổi thành thông điệp thân thiện với người dùng.
