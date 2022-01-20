# _*_coding:utf-8_*_
# 开发人：戴祥祥
# 开发时间：2021-08-07  11:20
# 文件名：test.py
class Solution:
    def isValid(self, s: str) -> bool:
        dic = {'{':'}', '[':']', '(':')'}
        stack = []
        try:
            for i in s:
                if i in  dic.keys():
                    stack.append(i)
                elif dic[stack.pop()] != i:
                    return False
            return True
        except:
            return False
a = Solution()
print(a.isValid('['))