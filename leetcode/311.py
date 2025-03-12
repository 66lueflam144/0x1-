from collections import defaultdict


class Solution:

    def twoSum(self, nums: list[int], target: int) -> list[int]:
        hashtable = dict()
        for i, num in enumerate(nums):
            print(f"i:{i}, num: {num}")
            if target - num in hashtable:
                print(f"========target-num: {target - num}\ni: {i}\nhashtable[{target-num}]")

                return [hashtable[target - num], i]
            hashtable[nums[i]] = i
        return []

    def strs_in_same_(self, str_list: list[str]):
        res = defaultdict(list) # 初始化默认值为list的字典
        for x in str_list:
            res["".join(sorted(x))].append(x) # 对每一个单词进行排序，作为key，同时将该单词作为value写入
            # print(f"res: {res}, x:{x}")
        return list(res.values()) # 返回值

    # def longestConsecutive(self, nums: list[int]) -> int:
    #
    #     """
    #     它只是简单地计算了排序后数组中相邻元素差值为1的次数，而没有考虑到连续序列的起始点。这样会导致重复计算，从而得到错误的结果。
    #     :param nums:
    #     :return:
    #     """
    #     sort_res = sorted(nums)  # 先进行排序
    #     print(sort_res)
    #     count = 1  # 一个数字也是连续1
    #
    #     if not nums:
    #         return 0
    #
    #     for i in range(0, len(sort_res) - 1):
    #         if sort_res[i + 1] - sort_res[i] == 1:
    #             print(f"Last one: {sort_res[i+1]}, this one: {sort_res[i]}")# 判断, 但是这个版本有个致命问题，只有专注了部分
    #             # 当遇到[9,1,4,7,3,-1,0,5,8,-1,6]这类，就会出现错误
    #             count += 1
    #
    #     return count

    def longestConsecutive(self, nums:list[int]) -> int:

        if not nums: # 对输入进行验证，验证的方面：是否为空，是否存在
            return []

        num_set = set(nums)
        longest_streak = 0

        for num in num_set: # 取一个
            if num - 1 not in num_set: # 如果不存在前一个，则当前的就是起点
                current_num = num
                current_streak = 1


            while current_num + 1 in num_set: # 在是起点的前提下，如果有后一个
                current_num += 1
                current_streak += 1

            longest_streak = max(longest_streak, current_streak) # 通过上一个代码知道存在好几段连续的所以选择里面最大的
        return longest_streak

    def moveZeroes(self, nums: list[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        non_at = 0  # 非0数所在位置
        n = len(nums)

        for i in range(n):
            if nums[i] != 0:
                nums[non_at] = nums[i]  # 如果不是0，就将这个数字往前移动
                non_at += 1  # 这个非0位置被占用，就前往下一个
        for u in range(non_at, n):
            nums[u] = 0  # 移动到最后的时候，剩下的位置就是0的位置

    def maxArea(self, height: list[int]) -> int:
        left = 0
        right = len(height) - 1

        res = 0

        while left < right:
            min_h = min(height[left], height[right])
            s_ = min_h * (right - left)
            res = max(res, s_)

            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
        return res

    def threeSum(self, nums: list[int]) -> list[list[int]]:
        """
        关于使用.sort()和sorted的一些：
        两者几乎相同，但是后者接收的输入是任何可迭代对象，返回新列表
        前者在原list上进行修改

        在这个题目中，为什么使用sorted(nums)失效，因为返回的新list没有被作为nums利用。写成nums = sorted(nums)就可以了。
        :param nums:
        :return:
        """
        # nums = sorted(nums) # 任意可迭代对象
        nums.sort() # 对list进行排序
        res = []
        n = len(nums)

        k = 0 # k作为指标，通过它找另外两个

        for k in range(n-2):
            # print(f"k: {k}, range: {n-2}")

            if nums[k] > 0: # 检查一：排序后是否第一个大于0，大于0则说明整个数组不存在 = 0的可能，结束循环
                break
            if k > 0 and nums[k - 1] == nums[k]: # 检查二：数组元素是否重复，重复的就不必再进行计算，快进到下一个k
                continue
            i = k + 1 # 右指针
            j = n - 1 # 左指针

            while i < j:
                s = nums[k] + nums[i] + nums[j]
                if s > 0:
                    j -= 1
                    while i < j and nums[j] == nums[j + 1]: # 去除重复的计算
                        j -= 1
                elif s < 0:
                    i += 1
                    while i < j and nums[i] == nums[i - 1]:
                        i += 1
                else:
                    res.append([nums[k], nums[i], nums[j]]) # 如果得到0结果，就添加到数组中
                    # 移动指针继续进行循环
                    i += 1
                    j -= 1
                    while i < j and nums[j] == nums[j + 1]:
                        j -= 1
                    while i < j and nums[i] == nums[i - 1]:
                        i += 1
        return res





if __name__ == '__main__':

    test_ = Solution()
    # nums = [2,7,11,15]
    # target = 17
    # test_.twoSum(nums, target)
    # words_list = ["eat", "nut", "ate", "his", "ish"]
    # x = test_.strs_in_same_(words_list)
    # print(x)
    # nums = [9,1,4,7,3,-1,0,5,8,-1,6]
    # test_.longestConsecutive(nums)

    a = [1, 2, 3]
