from typing import Tuple


class Rect:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def origin(self):
        return self.x, self.y

    def diagonal(self):
        return self.x + self.w, self.y + self.h

    def corners(self):
        return [self.origin(), self.diagonal()]

    def area(self):
        return self.w * self.h

    def translate(self, delta):
        self.x += delta[0]
        self.y += delta[1]

    def box(self):
        return self.origin()[0], self.origin()[1], self.diagonal()[0], self.diagonal()[1]

    def __str__(self):
        return f'Rect(origin: {self.origin()}, diagonal: {self.diagonal()})'

