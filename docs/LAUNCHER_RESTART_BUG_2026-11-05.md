# Launcher 缺陷报告：`restart` 报成功，但旧进程还在服务

**日期：** 2026-11-05
**报告人：** ComfyUI-MieNodes 插件开发者（agent 代写，可直接转交启动器开发）
**环境：** `C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\`，Windows，`ComfyUI启动器-CLI.cmd`（CLI 模式）
**严重度：** 高 —— `restart` 是"部署新代码后重新加载插件"的唯一手段，它报成功但没换进程，会让**一整个下午/晚上的渲染都在跑旧代码**，而且没有任何报错。

---

## 1. 复现（本次实录）

```
$ cmd //c "ComfyUI启动器-CLI.cmd" restart --json
添加 path_tools 子目录到 PATH: sox
启动工作目录(cwd): C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\ComfyUI
ComfyUI /system_stats 就绪，启动完成
{"stopped": false, "started": true, "ready": true, "elapsed_sec": 3.187, "pid": 75820, "port": 8188, ...}
```

字段解读（按 launcher 自己的语义）：

- `"stopped": false` —— **旧进程没被停掉**；
- `"started": true` —— launcher 确实拉起了新进程 75820；
- `"ready": true` —— **"就绪"是假的**：探针 8188 时回答的是旧进程。

新进程 75820 在 3 秒内就死了，`comfyui.log` 里写得很清楚：

```
[ERROR] Port 8188 is already in use on address 0.0.0.0. Please close the other application or use a different port with --port.
```

端口的真正持有者没变：

```
$ netstat -ano | grep ":8188.*LISTENING"
  TCP    0.0.0.0:8188    0.0.0.0:0    LISTENING    44288      <-- 旧进程，还在服务
```

而 launcher 自己也承认它管不到这个进程：

```
$ cmd //c "ComfyUI启动器-CLI.cmd" stop --json
{"stopped": false, "pid": null, "elapsed_sec": 0.0}
```

**`pid: null` = 它连"端口占用者"这条兜底链都没走，直接什么都没做。**

## 2. 人工兜底（目前唯一可靠的重启姿势）

```
$ netstat -ano | grep ":8188.*LISTENING"     # 找出端口持有者 pid
$ taskkill //PID 44288 //F                  # 直接杀掉（Git Bash 下 //PID //F 可用）
$ cmd //c "ComfyUI启动器-CLI.cmd" start --json
{"started": true, "pid": 72356, "ready": true, "elapsed_sec": 22.75, ...}
```

## 3. 为什么"ready: true"是假阳性

launcher 用 `GET /system_stats` 是否 200 来判就绪。8188 上有**任何一个** ComfyUI 在应答，这个探针就通过 —— 它分辨不出"应答的是我刚拉起的那个，还是上一班留下的那个"。重启之后旧进程还在监听，这个探针永远指向旧进程。

**更狠的是：探针通过 ⇒ 启动器认为成功 ⇒ 新进程即使已经因为端口占用而崩溃，也不会有任何人收到错误。**

按 `cli.md` 自己的约定，`restart` 的退出码 `0` 表示"**重启后服务就绪**"。本次它返回 0，而服务其实是**旧的**——退出码的语义被破坏了：调用方（脚本、agent、我）无法从返回值判断"代码是否真的换上了"。

## 4. 请求修的 5 点（按优先级）

1. **`restart` 必须确认"我拉起的进程就是端口上的那个"。**
   启动后 netstat 一次：监听 pid == 刚报的 pid 才算成功；不等就杀掉端口持有者重试一次，仍失败则**非 0 退出码 + 一行中文说明**（"重启未生效：端口 8188 仍被 pid X 持有"）。
2. **`stop` 的端口兜底链必须真的会走。**
   `cli.md` 自己写的是「按 pidfile + **端口回退链**停掉 ComfyUI」，但 pidfile 失准（进程不是 launcher 起的）时它判定"没在跑"，直接返回 `pid: null` 什么都不做——端口明明被 44288 占着。应当：pidfile 失准 → 查端口持有者 → 杀掉它 → 报告杀的是谁。
3. **"就绪"要探自己的进程，不要探端口。**
   最低成本：监听 pid == 启动 pid；更稳：比对 `comfyui.log` 里该 pid 的启动横幅时间戳。端口上另一个进程在应答 ≠ 你重启成功。
4. **子进程的 bind 失败要冒到 launcher 的返回值里。**
   现在 bind 失败只有 ComfyUI 自己那行 `[ERROR] Port 8188 is already in use`，launcher 依然返回 `started:true, ready:true`。
5. **建议给 `restart` 加一条"部署后自检"**：比较监听进程的启动时间 与 目标目录里最新文件的 mtime；旧进程在跑就直接报"代码未生效"。这正是本次事故的根因——我改了插件代码、restart 报成功、当晚所有渲染跑的是旧代码。

## 5. 影响面（这条链路上踩过的坑）

同一台机器上，这个"restart 报成功但没换进程"的坑**至少发生过 7 次**：2026-09-21、10-11（两次）、10-12、10-15、10-20、10-30、11-05。每次的代价都一样：以为部署成功了，实际跑的是旧代码，**只有靠人工 netstat + taskkill 才能恢复**。

对插件开发者来说这条链路是刚需：ComfyUI 只在启动时加载 custom_nodes，没有热重载，`restart` 失败 = 代码没生效。

## 6. 一句话总结

> `restart` 现在只保证"我 spawn 了一个进程"，不保证"服务我的进程被换掉了"。请把它改成**以端口监听 pid 为准**的成功判定，并把"没换掉"变成一个响亮的失败。
