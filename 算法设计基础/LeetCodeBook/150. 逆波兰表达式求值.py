# 后缀表达式   类似后序遍历
# "操作数① 操作数③ 运算符②"
# python3整除   //   int()
# 判断字符串是否是合理的整数   try except
#  求值方法
# 如果遇到数字就进栈
# 如果遇到操作符，就从栈顶弹出两个数字num2(栈顶)num1(栈中第二个元素) num1运算num2


class Solution:
    def evalRPN(self, tokens):
        """
        栈
        时间复杂度：O（N） 
        """
        stack= [] 
        for token in tokens:
            try:
                stack.append(int(token))
            except:
                num2=stack.pop() 
                num1=stack.pop()
                stack.append(self.evalute(num1,num2,token))


        return stack[0]


    def evalute(self, num1, num2, op):
        if op == "+":
            return num1+num2
        if op == "-":
            return num1-num2
        if op == "*":
            return num1*num2
        if op == "/":
            return int(num1/float(num2))
