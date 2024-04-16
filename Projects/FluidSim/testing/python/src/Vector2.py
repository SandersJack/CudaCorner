import math

class Vector2:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return Vector2(self.x / scalar, self.y / scalar)

    def __neg__(self):
        return Vector2(-self.x, -self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self):
        mag = self.magnitude()
        if mag != 0:
            return self / mag
        else:
            return Vector2()

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def angle(self, other):
        dot = self.dot(other)
        mag_self = self.magnitude()
        mag_other = other.magnitude()
        if mag_self == 0 or mag_other == 0:
            return 0
        return math.acos(dot / (mag_self * mag_other))