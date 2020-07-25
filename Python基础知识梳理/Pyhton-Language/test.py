import random
a = int(random.randint(1, 100))
print("猜测1-100的一个数字")
i = 1
while True:
    x = input('第%d次猜测，请输入一个数字：' % i)
    try:
        if type(eval(x)) == int:
            guessnum = int(x)
            if guessnum < a:
                print('小了！')
            elif guessnum > a:
                print("大了！")
            else:
                print("恭喜您猜到了，这个数是%d" % a)
                break
    except:
        print("输入无效")
    i += 1
