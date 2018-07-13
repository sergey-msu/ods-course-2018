class Node:
  def predict(self, x):
    pass


class InnerNode(Node):
  def __init__(self, pred, lnode, rnode):
    self.__pred = pred
    self.__lnode = lnode
    self.__rnode = rnode

  def predict(self, x):
    return self.__rnode.predict(x) if self.__pred(x) else self.__lnode.predict(x)


class LeafNode(Node):
  def __init__(self, value):
    self.__value = value

  @property
  def value(self):
    return self.__value

  def predict(self, x):
    return self.__value