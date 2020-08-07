# 查询任意年内所有周日
import parser
import datetime

a = input('请输入年份:')
# print(type(a))


def all_sundays(year):
    a0 = int(a)
    dt1 = datetime.date(a0, 1, 1)
    dt2 = datetime.date(a0, 12, 31)
    # print(dt1,dt2)
    dt = (dt2 - dt1).days
    '''zhou=(dt//7)#周数
    zhouji=dt1.isoweekday()#判断当前周几
    cha=7-zhouji
    td_cha=datetime.timedelta(days=cha)
    first_sunday=dt1+td_cha#第一个周日
    print(first_sunday)'''
    # td_cha7=datetime.timedelta(days=7)#7天周日期差
    # difference=0

    for i in range((dt2 - dt1).days + 1):
        # print(i)
        day = dt1 + datetime.timedelta(days=i)
        bianli = day.isoweekday()
        if bianli == 7:
            print(day)
        else:
            continue
all_sundays(a)
