# 吕明田的绿球识别作业

合并后的远端原来包含两份不同的提交，整理目录时分别保留：

| 文件 | 来源 | 本地结果 |
| --- | --- | --- |
| `main.cpp` | 原 `homework/week1-git-and-cpp/lvmingtian.cpp` | `result.mp4` |
| `legacy/main.cpp` | 原 `homework/week1-git-basics/lvmingtian/main.cpp` | `legacy/result.mp4` |

两份实现的检测逻辑不同，不能按同名文件覆盖。原来的空文件
`homework/lvmingtian` 已由现在的个人目录替代。

从仓库根目录运行，并先准备 `homework.mp4`。OpenCV 4 编译示例：

```bash
g++ -std=c++17 homework/lvmingtian/main.cpp -o /tmp/lvmingtian $(pkg-config --cflags --libs opencv4)
/tmp/lvmingtian

g++ -std=c++17 homework/lvmingtian/legacy/main.cpp -o /tmp/lvmingtian-legacy $(pkg-config --cflags --libs opencv4)
/tmp/lvmingtian-legacy
```

默认输出各自目录中的 `result.mp4`；视频只保留在本地，不提交 Git。
