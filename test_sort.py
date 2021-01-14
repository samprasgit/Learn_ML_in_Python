def test(a, *args):
    """a是一个普通传入的参数
    :args是一个非关键字星号参数
    """
    print('*args:{0}'.format(args))


a = [1, 2, 3]
test(4, *a)
