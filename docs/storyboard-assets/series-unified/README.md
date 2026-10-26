# 千层套路：统一系列版

三集共用定版Claude娘、小D的声音和同一夜晚书房。003保留夜晚。开场语言随每集台词，不作为系列验收条件；封禁收尾、服装、面容、房间和坐姿统一。原始分镜和资产保留。

## 使用

1. 导入三个 `Claude的千层套路00N-系列版.json` 分镜（不要把series-manifest.json作为分镜导入）。
2. V3：character_1接“小D-千层套路声音”，character_2接“Claude娘-千层套路定版”，scene_1接“千层套路书房-夜晚”。
3. Asset→Slots：picture_1接H3 ref_image_0（Claude娘），picture_2接ref_image_1（客厅）。旁白图片自动排除。
4. audio_1接H3 ref_audio_0（小D），audio_2接ref_audio_1（Claude娘）。声音模式reference；Generation Profile仍用Generate audio，参考声音与Source Timeline配音不是同一个通道。
5. 002生成后执行本次QA中的finalize_002.py，确定性显示 `Hello，I have a problem` 并高亮全角中文逗号；H3负责表演和对白。

参考声音提供身份条件，是否听感一致仍需成片验收。此模式最多3个人物，每个角色都必须有参考声音，缺失会报错。

镜头只使用一个生成片段，原坐姿说完封禁台词。003在午饭回复后先疑惑，再合嘴邪魅一笑，再说封禁台词。
