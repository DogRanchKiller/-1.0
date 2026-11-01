"""正式网站的静态页面（需求文档第 8 节“正式网站（静态页面）实际做法”，第 9 节接口 4）。

几个文件的分工：
  data.py     读数据：数据库（经过取数层，用途“显示”）、预测存档 predictions 表、data/review/ 的对账结果
  common.py   几页共用的小工具：北京时间、百分比、安全的文件名、页面外框、内嵌 SVG 图（热力图、折线图）
  match.py    单场页 match/<match_id>.html
  pages.py    首页 index.html、球队页 team/<球队>.html、战绩页 record.html
整站由 src/jobs/build_site.py 生成到 data/site/。只读数据，不出预测、不跑模型、不联网；只用 HTML + CSS，没有脚本。
"""
