import requests
start_pn = 0  # 初始 pn 值
str = ('https://image.baidu.com/search/acjson?tn=resultjson_com&logid=7735428982766424353&ipn=rj&ct=201326592&is=&fp'
       '=result&fr=&word=%E4%BA%95%E5%9C%88%E7%A0%B4%E6%8D%9F&queryWord=%E4%BA%95%E5%9C%88%E7%A0%B4%E6%8D%9F&cl=2&lm'
       '=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=0&istype=2'
       f'&qc=&nc=1&expermode=&nojc=&isAsync=&pn={start_pn}&rn=30&gsm=1e&1711546213081=')
headers = {
    'Host': 'image.baidu.com',
    'Cookie': 'BDqhfp=%E4%BA%95%E7%9B%96%E5%AE%8C%E5%A5%BD%E5%9B%BE%E7%89%87%26%26NaN-1undefined%26%260%26%261; '
              'BIDUPSID=D02A8140B4BB076983DDEC1ED5BA460A; PSTM=1709897174; ',
    'Referer': ('https://image.baidu.com/search/acjson?tn=resultjson_com&logid=7735428982766424353&ipn=rj&ct=201326592&is=&fp'
       '=result&fr=&word=%E4%BA%95%E5%9C%88%E7%A0%B4%E6%8D%9F&queryWord=%E4%BA%95%E5%9C%88%E7%A0%B4%E6%8D%9F&cl=2&lm'
       '=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=0&istype=2'
       f'&qc=&nc=1&expermode=&nojc=&isAsync=&pn={start_pn}&rn=30&gsm=1e&1711546213081='),
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 '
                  'Safari/537.36 Edg/121.0.0.0 ',
}
number = 1

for page in range(0, 20):
    str = ('https://image.baidu.com/search/acjson?tn=resultjson_com&logid=7735428982766424353&ipn=rj&ct=201326592&is=&fp'
       '=result&fr=&word=%E4%BA%95%E5%9C%88%E7%A0%B4%E6%8D%9F&queryWord=%E4%BA%95%E5%9C%88%E7%A0%B4%E6%8D%9F&cl=2&lm'
       '=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=0&istype=2'
       f'&qc=&nc=1&expermode=&nojc=&isAsync=&pn={start_pn + page * 30}&rn=30&gsm=1e&1711546213081=')
    url = str
    response = requests.get(url=url, headers=headers)
    json_data = response.json()
    data_list = json_data['data']
    for data in data_list[:-1]:
        fromPageTitleEnc = data['fromPageTitleEnc']
        middleURL = data['middleURL']
        print(fromPageTitleEnc, middleURL)
        img_data = requests.get(middleURL).content
        with open(f'img/{number}.jpg', mode='wb') as f:
            f.write(img_data)
        number += 1
