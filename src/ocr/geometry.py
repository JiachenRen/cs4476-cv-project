from typing import Tuple


class Rect:
    origin: Tuple[int, int]
    diagonal: Tuple[int, int]

    def __init__(self, x, y, w, h):
        self.origin = (x, y)
        self.diagonal = (x + w, y + h)

    def width(self):
        return self.diagonal[0] - self.origin[0]

    def height(self):
        return self.diagonal[1] - self.origin[1]

    def corners(self):
        return [self.origin, self.diagonal]

    def __str__(self):
        return f'Rect(origin: {self.origin}, diagonal: {self.diagonal})'

