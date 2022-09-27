# <center>🎉AI文章生成</center>



AI文章生成系统基于**pytorch**框架使用机器学习中的**CPM模型**，来实现**根据标题智能生成文章**的功能。

<center><img src="https://img-blog.csdnimg.cn/9ff52fa6abd04adaa17a9d398476abb5.gif" width="60%"></center>



## 😇项目灵感

项目的灵感来源于客户的需求🤑，首先**我们团队**之前并**没有接触😱**过通过**标题生成文章**的这类自然语言方面的项目，**但是做过**类似的项目，**通过GPT2模型根据文章来生成标题**。然后，就想着直接**逆过程**能不能行，就是将**输入**和**输出换**了一个**方向**，相当于**原来**是**文章**作为**输入**，**标题**作为**输出**对模型进行训练，换成**现在** **标题**作为**输入**，**文章**作为**输出**对模型进行训练。我们尝试训练了之后，训练出来的**模型**进行测试发现**效果不是很好**，这样生成的文章不仅读不通顺，而且存在许多的重复语句，根本达不到客户的需求。经历了这次失败的经验之后，我们也在网上查阅了很多相关的资料，**换**了一种更加成熟、效果可能更好的**模型**，**CPM(Chinese Pretrained Models)模型**👻。

清源 CPM(Chinese Pretrained Models)是北京智源人工智能研究院和清华大学研究团队合作开展的大规模**预训练模型开源计划**，清源计划是以**中文为核心**的大规模**预训练模型**。首期开源内容包括预训练中文语言模型和预训练知识表示模型，可广泛应用于中文自然语言理解、生成任务以及知识计算应用，所有模型**免费向学术界和产业界开放下载，供研究使用**。**[[CPM官网](https://cpm.baai.ac.cn/)] [[模型下载](https://cpm.baai.ac.cn/download.html)] [[技术报告](https://arxiv.org/abs/2012.00413)] [[CPM微调源码](https://github.com/TsinghuaAI/CPM-1-Finetune)]  [[CPM生成源码](https://github.com/TsinghuaAI/CPM-1-Generate)]**

## 📽项目演示

#### （1）🍺根据标题生成文章
<center><img src="https://img-blog.csdnimg.cn/9ff52fa6abd04adaa17a9d398476abb5.gif" width="60%"></center>

#### （2）🍻批量文本生成
<center><img src="https://img-blog.csdnimg.cn/7fefcfdcec50409bb5e58777ec31fc5d.gif" width="65%"></center>

## 🎯运行环境

#### （1）🍋前期准备
本项目整体代码是**基于python语言**实现的，所以我们需要准备好以下**开发工具**：
- **anaconda** - anaconda的安装使用可以参考博客[anaconda详细安装使用教程](https://blog.csdn.net/xc_zhou/article/details/82715612)。
- **pycharm** - pycharm安装[官网](https://www.jetbrains.com/pycharm/)的社区版就够用了，pycharm的安装可以参考博客[pycharm从安装到全副武装，学起来才嗖嗖的快，图片超多，因为过度详细！](https://blog.csdn.net/weixin_46211269/article/details/119934323)

#### （2）🍍具体配置
- 1.打开Anaconda Prompt,输入
```bash
conda create -n pytorch python=3.6
```
创建一个**python版本为3.6**的虚拟环境。
- 2.进入到pytorch虚拟环境
```bash
conda activate pytorch
```
- 3.在pytorch虚拟环境安装项目运行**所必须的第三方库**
    -  安装pytorch，pytorch[下载官网](https://pytorch.org/)，安装时大家**注意自己cuda的版本**，**没有显卡**的可以**安装cpu版**。pytorch的安装可以参考博客[PyTorch 最新安装教程（2021-07-27）](https://blog.csdn.net/qq_46092061/article/details/119153893)
    ```bash
    # CUDA 10.2
    conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
    ```
    
    - 安装transformers
    ```bash
    pip install transformers=4.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
   - 安装jieba分词库
	```bash
	pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple
	```
   - 安装ttkbootstrap
	```bash
	pip install ttkbootstrap -i https://pypi.tuna.tsinghua.edu.cn/simple
	```
	
- 4.**环境配置配置好了**之后，使用**pycharm打开项目**，并将pycharm链接刚刚配好的pytorch虚拟环境的解释器(python.exe)，**运行main文件**即可🎷。

## ☕请我们喝卡布奇诺

如果本仓库对你有帮助，可以请作者喝杯卡布奇诺☜(ﾟヮﾟ☜)

<center><img src="https://user-images.githubusercontent.com/112611204/192464009-5ecf272b-c818-4fff-9569-7f3d42d5042b.png" width="40%"></center>





