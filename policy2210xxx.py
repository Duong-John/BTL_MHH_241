import numpy as np
from policy import Policy

class Policy2210xxx(Policy):
    def __init__(self, policy_id: int = 1):
        """
        Khởi tạo chính sách cắt sản phẩm.

        Args:
        policy_id (int): ID của chính sách (phải là 1).
        """
        assert policy_id == 1, "Policy ID phải là 1"

    def get_action(self, observation: dict, info: dict) -> dict:
        """
        Lấy hành động cắt sản phẩm dựa trên quan sát.

        Args:
        observation (dict): Dữ liệu quan sát bao gồm 'stocks' và 'products'.
        info (dict): Thông tin bổ sung.

        Returns:
        dict: Hành động cắt sản phẩm hoặc None nếu không tìm được vị trí cắt.
        """
        stocks = observation['stocks']
        products = observation['products']

        # Chuyển đổi products thành danh sách và sắp xếp theo thứ tự kích thước giảm dần
        products = sorted(products, key=lambda x: x['size'][0]*x['size'][1], reverse=True)

        for stock_idx, stock in enumerate(stocks):
            for product in products:
                # Tìm vị trí cắt tối ưu
                pos_x, pos_y = self.find_optimal_position(stock, product['size'])
                if pos_x != -1 and pos_y != -1:
                    self.cut_product(stock, product['size'], pos_x, pos_y, product)
                    product['quantity'] -= 1
                    return {
                        "stock_idx": stock_idx,
                        "size": product['size'],
                        "position": (pos_x, pos_y)
                    }
        return None

    def find_optimal_position(self, stock: np.ndarray, product_size: tuple) -> tuple:
        """
        Tìm vị trí cắt tối ưu cho sản phẩm.

        Args:
        stock (np.ndarray): Stock hiện tại.
        product_size (tuple): Kích thước sản phẩm.

        Returns:
        tuple: Vị trí cắt tối ưu (x, y) hoặc (-1, -1) nếu không tìm được.
        """
        max_width, max_height = stock.shape
        for x in range(max_width - product_size[0] + 1):
            for y in range(max_height - product_size[1] + 1):
                if self.can_cut_product(stock, product_size, x, y):
                    return x, y
        return -1, -1

    def can_cut_product(self, stock: np.ndarray, product_size: tuple, pos_x: int, pos_y: int) -> bool:
        """
        Kiểm tra nếu sản phẩm có thể cắt vào vị trí (pos_x, pos_y) trong stock.

        Args:
        stock (np.ndarray): Stock hiện tại.
        product_size (tuple): Kích thước sản phẩm.
        pos_x (int): Tọa độ x.
        pos_y (int): Tọa độ y.

        Returns:
        bool: True nếu sản phẩm có thể cắt, False ngược lại.
        """
        pw, ph = product_size
        return np.all(stock[pos_x:pos_x+pw, pos_y:pos_y+ph] == -1)

    def cut_product(self, stock: np.ndarray, product_size: tuple, pos_x: int, pos_y: int, product: dict) -> None:
        """
        Cắt sản phẩm vào stock.

        Args:
        stock (np.ndarray): Stock hiện tại.
        product_size (tuple): Kích thước sản phẩm.
        pos_x (int): Tọa độ x.
        pos_y (int): Tọa độ y.
        product (dict): Thông tin sản phẩm.
        """
        pw, ph = product_size
        product_id = product.get('id', -1)
        stock[pos_x:pos_x+pw, pos_y:pos_y+ph] = product_id