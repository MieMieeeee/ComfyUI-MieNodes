# 内置资产（bundled assets）

放在这个目录里的资产**随插件包一起分发**：用户装好插件就能在「加载资产」/「加载分镜」的下拉里
直接选到它们，不用先导入、不用配置路径。

## 目录结构

```
bundled_assets/
  characters/<角色名>/character.json   # + images/ 与 audio/（和用户资产库同结构）
  scenes/<场景名>/scene.json           # + images/
  storyboards/<分镜名>/storyboard.json
```

与用户资产库**完全同构**，所以复制进来即可用。

## 优先级：用户库优先

- 用户资产库里同名的资产**优先**于内置副本——用户可以随时用自己的版本替换内置的。
- 用户库里没有的名字，才回落到内置目录。
- 「保存资产 / 保存分镜」永远写进用户资产库，**不会**改动内置目录（内置目录只读）。
- 下拉列表里两者都会列出，同名只出现一次。

## 怎么放进来

```bash
# 从本机资产库复制（默认源：<插件>/asset_library）
python scripts/bundle_assets.py --list
python scripts/bundle_assets.py characters/Claude娘 characters/Claude娘-日式赛璐璐
python scripts/bundle_assets.py --all-characters "Claude娘*"
python scripts/bundle_assets.py --all-storyboards
python scripts/bundle_assets.py --dry-run characters/Claude娘    # 只看会复制什么
```

## 注意

- **包体**：每个角色资产 ≈2–3 MB（一张 PNG + 一段 WAV）。内置七个 Claude娘 变体约 17 MB，
  Registry 包体会同比变大；分享前请确认可接受。
- **只放可公开分发的内容**：角色图、声线样音、分镜 JSON 都随包分发，等于公开发布。
- 复制进来的资产若与分镜资产的角色名对不上，运行时才会报"找不到角色"——用
  `python scripts/validate_storyboard_asset.py` 逐个校验分镜，用
  `python scripts/import_storyboard_assets.py --help` 确认角色名对得上。
