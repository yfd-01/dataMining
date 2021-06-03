# encoding: utf-8
import time
import math
import itertools
import numpy as np

# function


def check_ele_exist(elem, line):
    """
    判断元素集合在一行中是否出现
    :param elem: 作为子集
    :param line: 作为父集
    :return: ``True``出现，``False``未出现
    """
    return True if set(elem) <= set(line) else False

# class


class Apriori:
    """
        Apriori类
        基于Apriori关联规则挖掘算法，导入数据文件，用于得到不同项数的频繁项集

        :param ls:
            被挖掘的数据集合（可以通过ls参数给出被挖掘数据集合，而不用数据文件导入）

        :param min_support_rate:
            最小的支持阈值

        :param min_support_degree:
            最小的支持度

        :param data_file_name:
            导入的数据集合文件名

        :param out_file_name:
            导出的结果数据文件名

        :param silence_mode:
            是否在控制台输出中间结果
            ``True`` 不输出
            ``False`` 输出
    """

    def __init__(
        self,
        ls=None,
        min_support_rate=0.5,
        min_support_degree=-1,
        data_file_name='',
        out_file_name='',
        silence_mode=False
    ):
        self.ls = ls
        self.data_file_name = data_file_name
        self.out_file_name = out_file_name
        self.min_support_rate = min_support_rate
        self.min_support_degree = min_support_degree

        if isinstance(ls, list) or isinstance(ls, np.ndarray):
            #ls为列表或numpy数组类型,直接采用
            self.ls = ls
        else:
            #ls需从文件导入
            self.ls = []
            self.file_load(self.data_file_name)

        self.start_time = time.time()   #记录开始时间
        self.checked_time = time.time() #记录标记时间(用于得到不同项集分别用时)
        self.end_time = None
        self.out_file_time_sum = 0      #统计输出到文件时间

        self.silence_mode = silence_mode

    def get_frequent_items(self):
        """
        得到所有不同项数的频繁项集
        """
        self.get_k_fi(self.get_1_fi())

    def file_load(self, data_file_name):
        """
        加载数据文件

        :param data_file_name: 数据文件名字
        """
        with open(data_file_name, 'r') as data_file:
            for line in data_file.readlines():
                #line.split(' ')包含换行符所以去掉最后一个字符 => line.split(' ')[: -1]
                #split后的列表中的每个元素是字符串，用map映射对每个元素进行字符串到数字转换 => map(func, line.split(' ')[: -1])
                #func用lambda表达式 => func = lambda n: int(n)
                #将map返回值对象转换为list对象 => list(Object(Map))
                self.ls.append(list(map(lambda n: int(n), line.split(' ')[: -1])))

        self.min_support_degree = math.ceil(self.min_support_rate * len(self.ls))   #通过阈值计算最低支持度（向上取整）

    def file_write(self, text, to_console=True):
        """
        写入结果文件

        :param text: 写入文件的一行信息
        :param to_console: 是否将写入文件的信息输出在控制台
        """
        with open(self.out_file_name, 'a', encoding="utf-8") as out_file:
            out_file.writelines(text)

            if not self.silence_mode and to_console:    #非安静模式且需要输出到控制台
                print(text)

    def get_1_fi(self):
        """
        得到频繁1项集

        :return: 返回频繁1项集结果（用于更多项数的频繁项集的挖掘）
        """
        self.file_write("--- 当前阈值设定为: %.2f ---\n" % self.min_support_rate)

        fi_ = []    #用于保存频繁1项集

        max_ = np.array(self.ls, dtype=list).max()  #获取最大的元素
        if isinstance(max_, list):  #如果获取的是最大的一行
            max_ = max(max_)

        arr = [0] * (max_+1) #统计每个元素出现次数，第几个元素用下标进行表示

        for i in self.ls:   #遍历数据集合，每次以元素值作为下标进行加1
            for j in i:
                arr[int(j)] += 1

        for i, e in enumerate(arr):
            if e >= self.min_support_degree:    #将不低于支持度的放入结果列表
                fi_.append({i: e})  #结果列表每个元素格式为： {(n1, n2, ..., nk): supported_degree} => {tuple: int}

        t_time_s = time.time()  #记录写入文件开始时间
        if len(fi_):
            self.file_write("--- 频繁1项集\n", to_console=False)

        for i in fi_:
            self.file_write("%d: %d\n" % (*i.keys(), *i.values()), to_console=False)
        t_time_e = time.time()  #记录写入文件结束时间
        self.out_file_time_sum += t_time_e - t_time_s   #累加写入时间

        if not self.silence_mode:   #非安静模式下将结果输出控制台
            print("频繁1项集, 其个数为: %d" % len(fi_))
            print(fi_)

        return fi_

    def get_k_fi(self, last_fi, last_fi_l=None, k=2):
        """
        得到频繁k项集
        :param last_fi: 上一个频繁项集，即第k-1频繁项集（包含支持度）
        :param last_fi_l: 上一个频繁项集的所有对象集合（不包含支持度）
        :param k: 表明正在挖掘第几频繁项集
        """
        if len(last_fi) == 1:   #如果频繁k-1项集的元素只有一个了，一定没有频繁k项集，该阈值下的挖掘结束
            self.end_time = time.time() #标记结束时间
            return

        if k == 2:  #进行频繁2项集时，last_fi_l是没有提供的，需要单独求
            last_fi_l = []
            for i in last_fi:
                last_fi_l.append(set(i.keys()))

        fi_k_l = [] #存储频繁k项集（不包括支持度）
        fi_k = []   #存储频繁k项集（包括支持度）
        tmp_k_l = []   #频繁k项集的候补（里面的元素有可能会进入频繁k项集）
        for i, j in itertools.combinations(range(len(last_fi_l)), 2):   #把频繁k-1项集的元素进行组合
            t_e = sorted(set(last_fi_l[i]) | set(last_fi_l[j]))     #两个元素求并集然后排序
            if len(t_e) == k and t_e not in tmp_k_l:    #并集后的新元素是k的长度且没有在频繁k项集的候补
                tmp_k_l.append(t_e)

        arr = [0] * (len(tmp_k_l) + 1)  #同get_1_fi函数
        for line in self.ls:    #遍历ls每一行
            for i, e in enumerate(tmp_k_l): #在一行中对频繁k项集的候补中所有元素进行检测
                arr[i+1] += 1 if check_ele_exist(e, line) else 0

        for i, e in enumerate(arr[1:]): #同get_1_fi函数
            if e >= self.min_support_degree:    #同get_1_fi函数
                fi_k_l.append(tmp_k_l[i])
                fi_k.append({tuple(tmp_k_l[i]): e})

        if not len(fi_k):   #频繁k项集一个元素都没有
            self.end_time = time.time()
            return
        else:
            t_time_s = time.time()  #记录写入文件开始时间
            self.file_write("--- 频繁%d项集\n" % k, to_console=False)

            for i in fi_k:  #遍历频繁k项集写入文件
                self.file_write("%s: %d\n" % (*i.keys(), *i.values()), to_console=False)
            t_time_e = time.time()  #记录写入文件结束时间
            self.out_file_time_sum += t_time_e - t_time_s   #累加写入时间

            if not self.silence_mode:   #非安静模式下将结果输出控制台
                print("频繁%d项集, 其个数为: %d, 用时: %s" % (k, len(fi_k), time.time()-self.checked_time))
                print(fi_k)
            self.checked_time = time.time()     #标记时间更新

            self.get_k_fi(fi_k, fi_k_l, k+1)    #递归进入频繁k+1项集


if __name__ == "__main__":
    support_rate_arr = [0.25, 0.20, 0.15, 0.10, 0.05]   #多个阈值设定列表
    for r in support_rate_arr:
        #初始化Apriori对象
        foo = Apriori(min_support_rate=r, data_file_name="mushroom.dat", out_file_name="out.txt", silence_mode=False)
        foo.get_frequent_items()    #获取所有频繁项集
        foo.file_write("--- 阈值为%.2f,一共耗时%.2f秒 (已去掉输出到文件时间)\n\n" % (r, foo.end_time - foo.start_time - foo.out_file_time_sum))
