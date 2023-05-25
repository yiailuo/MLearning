import numpy as np


def fun(x1, x2):
    return 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)


X1s = np.linspace(-2.9, 12.0, 100)
X2s = np.linspace(4.2, 5.7, 100)
np.random.seed(0)  # 令随机数种子=0，确保每次取得相同的随机数

# 初始化原始种群
x1_population = np.random.uniform(-2.9, 12.0, 10)  # 在[-2.9,12.0)上以均匀分布生成10个浮点数，做为初始种群
x2_population = np.random.uniform(4.2, 5.7, 10)  # 在[4.2,5.7]上以均匀分布生成10个浮点数，做为初识种群
for x1_pop, x2_pop, fit in zip(x1_population, x2_population, fun(x1_population, x2_population)):
    print("x1=%5.2f, x2=%5.2f, fit=%.2f" % (x1_pop, x2_pop, fit))


# 对x1进行编码
def x1_encode(population, _min=-2.9, _max=12.0, scale=2 ** 21, binary_len=21):  # population必须为float类型，否则精度不能保证
    # 标准化，使所有数据位于0和1之间,乘以scale使得数据间距拉大以便用二进制表示
    normalized_data = (population - _min) / (_max - _min) * scale
    # 转成二进制编码
    binary_data = np.array([np.binary_repr(x, width=binary_len)
                            for x in normalized_data.astype(int)])
    return binary_data


# 对x2进行编码
def x2_encode(population, _min=4.2, _max=5.7, scale=2 ** 18, binary_len=18):  # population必须为float类型，否则精度不能保证
    # 标准化，使所有数据位于0和1之间,乘以scale使得数据间距拉大以便用二进制表示
    normalized_data = (population - _min) / (_max - _min) * scale
    # 转成二进制编码
    binary_data = np.array([np.binary_repr(x, width=binary_len)
                            for x in normalized_data.astype(int)])
    return binary_data


# 对两个编码进行合并
def encode(population_x1, population_x2):
    chroms_x1 = x1_encode(population_x1)
    chroms_x2 = x2_encode(population_x2)
    binary_data = np.char.add(chroms_x1, chroms_x2)
    return binary_data


chroms = encode(x1_population, x2_population)  # 染色体英文(chromosome)

for x1_pop, x2_pop, chrom, fit in zip(x1_population, x2_population, chroms, fun(x1_population, x2_population)):
    print("x1=%.2f, x2=%.2f, chrom=%s, fit=%.2f" % (x1_pop, x2_pop, chrom, fit))


def x1_decode(popular_gene, _min=-2.9, _max=12.0, scale=2 ** 21):  # 先把x从2进制转换为10进制，表示这是第几份
    # 乘以每份长度（长度/份数）,加上起点,最终将一个2进制数，转换为x轴坐标
    return np.array([(int(x, base=2) / scale * 14.9) + _min for x in popular_gene])


def x2_decode(popular_gene, _min=4.2, _max=5.7, scale=2 ** 18):
    return np.array([(int(x, base=2) / scale * 1.5) + _min for x in popular_gene])


def decode(chroms_v):
    chroms_x1 = []
    chroms_x2 = []
    for arr in chroms_v:
        temp_x1 = arr[:21]
        temp_x2 = arr[21:]
        chroms_x1.append(temp_x1)
        chroms_x2.append(temp_x2)
    dechroms_x1 = x1_decode(chroms_x1)
    dechroms_x2 = x2_decode(chroms_x2)
    return dechroms_x1, dechroms_x2


# 将编码后的chroms解码并代入到适应度函数中
# 得到结果后进行个体评价
dechroms_x1, dechroms_x2 = decode(chroms)
fitness = fun(dechroms_x1, dechroms_x2)
for x1_pop, x2_pop, chrom, fit in zip(x1_population, x2_population, chroms,
                                      fitness):
    print("x1=%5.2f, x2=%5.2f, chrom=%s,, fit=%.2f" %
          (x1_pop, x2_pop, chrom, fit))
fitness = fitness - fitness.min() + 0.000001  # 保证所有的都为正
print("fitness-fitness.min()=", fitness)


def Select_Crossover(chroms, fitness, prob=0.6):  # 选择和交叉
    probs = fitness / np.sum(fitness)  # 各个个体被选择的概率
    probs_cum = np.cumsum(probs)  # 概率累加分布
    each_rand = np.random.uniform(size=len(fitness))  # 得到10个随机数，0到1之间
    # 轮盘赌，根据随机概率选择出新的基因编码
    # 对于each_rand中的每个随机数，找到被轮盘赌中的那个染色体
    newX = np.array([chroms[np.where(probs_cum > rand)[0][0]]
                     for rand in each_rand])
    # 繁殖，随机配对（概率为0.6)
    # 6这个数字怎么来的，根据遗传算法，假设有10个数，交叉概率为0.6，0和1一组，2和3一组。。。8和9一组，每组扔一个0到1之间的数字
    # 这个数字小于0.6就交叉，则平均下来应有三组进行交叉，即6个染色体要进行交叉
    pairs = np.random.permutation(
        int(len(newX) * prob // 2 * 2)).reshape(-1, 2)  # 产生6个随机数，乱排一下，分成二列
    center = len(newX[0]) // 2  # 交叉方法采用最简单的，中心交叉法
    for i, j in pairs:
        # 在中间位置交叉
        x, y = newX[i], newX[j]
        newX[i] = x[:center] + y[center:]  # newX的元素都是字符串，可以直接用+号拼接
        newX[j] = y[:center] + x[center:]
    return newX


chroms = Select_Crossover(chroms, fitness)

dechroms = decode(chroms)
fitness = fun(dechroms_x1, dechroms_x2)


# 输入一个原始种群1，输出一个变异种群2  函数参数中的冒号是参数的类型建议符，告诉程序员希望传入的实参的类型。函数后面跟着的箭头是函数返回值的类型建议符，用来说明该函数返回的值是什么类型。


def Mutate(chroms: np.array):
    prob = 0.1  # 变异的概率
    clen = len(chroms[0])  # chroms[0]="111101101 000010110"    字符串的长度=18
    m = {'0': '1', '1': '0'}  # m是一个字典，包含两对：第一对0是key而1是value；第二对1是key而0是value
    newchroms = []  # 存放变异后的新种群
    each_prob = np.random.uniform(size=len(chroms))  # 随机10个数

    for i, chrom in enumerate(chroms):  # enumerate的作用是整一个i出来
        if each_prob[i] < prob:  # 如果要进行变异(i的用处在这里)
            pos = np.random.randint(clen)  # 从18个位置随机中找一个位置，假设是7
            # 0~6保持不变，8~17保持不变，仅将7号翻转，即0改为1，1改为0。注意chrom中字符不是1就是0
            chrom = chrom[:pos] + m[chrom[pos]] + chrom[pos + 1:]
        newchroms.append(chrom)  # 无论if是否成立，都在newchroms中增加chroms的这个元素
    return np.array(newchroms)  # 返回变异后的种群


newchroms = Mutate(chroms)

# 上述代码只是执行了一轮，这里反复迭代
np.random.seed(0)  #
x1_population = np.random.uniform(-2.9, 12.0, 100)  # 这次多找一些点
x2_population = np.random.uniform(4.2, 5.7, 100)
chroms = encode(x1_population, x2_population)

for i in range(1000):
    dechroms_x1, dechroms_x2 = decode(chroms)
    fitness = fun(dechroms_x1, dechroms_x2)
    fitness = fitness - fitness.min() + 0.000001  # 保证所有的都为正
    newchroms = Mutate(Select_Crossover(chroms, fitness))
    chroms = newchroms

dechroms_x1, dechroms_x2 = decode(chroms)
fitness = fun(dechroms_x1, dechroms_x2)
print("最大值：", np.max(fitness))
