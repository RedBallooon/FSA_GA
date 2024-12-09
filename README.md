# Tối ưu hóa thuật toán Flamingo Search kết hợp với thuật toán di truyền (FSA-GA)

## Giới thiệu

Dự án này thực hiện thuật toán **Flamingo Search** kết hợp với thuật toán **di truyền** (FSA-GA) để giải quyết các bài toán tối ưu hóa hàm số. FSA-GA là một phương pháp lai, kết hợp sức mạnh tìm kiếm toàn cục của **Flamingo Search** với khả năng khai thác không gian tìm kiếm hiệu quả của **thuật toán di truyền**.

---

## Phương pháp

### Flamingo Search Algorithm (FSA)
- **Nguồn cảm hứng:** Dựa trên hành vi kiếm ăn của chim hồng hạc.
- **Đặc điểm chính:**
  - **Di chuyển đàn:** Tập trung vào các khu vực tiềm năng.
  - **Tìm kiếm ngẫu nhiên:** Đảm bảo tính đa dạng của các giải pháp.
  - **Hội tụ nhanh:** Di chuyển về nguồn thức ăn tốt nhất.

### Genetic Algorithm (GA)
- **Nguồn cảm hứng:** Dựa trên chọn lọc tự nhiên và di truyền học.
- **Đặc điểm chính:**
  - **Lai ghép (Crossover):** Tạo ra thế hệ giải pháp mới.
  - **Đột biến (Mutation):** Tăng khả năng khám phá không gian tìm kiếm.
  - **Chọn lọc (Selection):** Giữ lại những giải pháp tối ưu nhất.

### Sự kết hợp (FSA-GA)
- **Ưu điểm:** 
  - Tận dụng FSA để khởi tạo các giải pháp có tiềm năng cao.
  - Sử dụng GA để cải thiện các giải pháp này qua nhiều thế hệ.

---

## Các hàm mục tiêu

FSA-GA được thử nghiệm trên **9 hàm mục tiêu chuẩn** (F1 đến F9), bao gồm:
1. Các hàm phi tuyến tính phức tạp.
2. Các hàm với không gian tìm kiếm có số chiều lớn.

---

## Kết quả thực nghiệm

- **So sánh hiệu suất:**
  - **FSA-GA** vượt trội so với các thuật toán FSA và GA đơn lẻ.
  - Đạt được **tốc độ hội tụ cao hơn** và **giá trị tối ưu tốt hơn**.
- **Đặc biệt:** Trong các bài toán tối ưu hóa với không gian nhiều chiều, FSA-GA thể hiện hiệu quả rõ rệt.


## Cách sử dụng

### 1. Cài đặt các thư viện
Bạn cần cài đặt các thư viện sau: 
pip install numpy matplotlib

### 2. Chạy thử nghiệm
# Import thư viện
from fsa_ga import run_experiment

# Chạy thuật toán với không gian tìm kiếm 10 chiều
run_experiment(dim=10)

### 3. Tùy chỉnh tham số
Số lượng cá thể: population_size
Số thế hệ: num_generations
Tỉ lệ lai ghép: crossover_rate
Tỉ lệ đột biến: mutation_rate

