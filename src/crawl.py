# -*- coding: utf-8 -*-
import os
from icrawler.builtin import GoogleImageCrawler


def crawl(
        dir_path,
        super_concept,
        city_name,
        keyword,
        num_per_keyword,
):
    '''
    crawl and save images from the web searched with specified `keyword` using Google search engine
    '''

    if city_name == 'nyc':
        city_name = 'new york'

    dirc_path = os.path.join(
        dir_path, 'data', 'original', super_concept, city_name,
        keyword.replace(' ', '_') + '_' + str(num_per_keyword))

    if not os.path.isdir(dirc_path):
        os.makedirs(dirc_path)

    # else:
    #     i = input('dirc_path {} already exists, skip crawling images of \'{}\' with keyword \'{}\' or not ? (y or n) >'.
    #         format(dirc_path, city_name, keyword))
    #     if i == 'y':
    #         print('continue to crawl images of {} with keyword \'{}\' ...'.
    #               format(city_name, keyword))
    #     else:
    #         print('stopping crawling images of {} with keyword \'{}\' ...'.
    #               format(city_name, keyword))
    #         return

    storage_path = dirc_path.replace('new york', 'nyc')
    crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        storage={'root_dir': storage_path})
    filters = dict(
        size='medium',
        color='color',
        lisence='noncommercial,modify',
        date=((2010, 1, 1), None),
    )

    keyword = city_name + ' ' + keyword
    print('\n----------')
    print('keyword used for search: `{}`'.format(keyword))
    print('---------------')
    crawler.crawl(
        keyword=keyword,
        filters=filters,
        max_num=num_per_keyword,
        min_size=(200, 200),
        max_size=None,
        file_idx_offset='auto',
    )


if __name__ == '__main__':
    dir_path = os.path.join('c:\\', 'Users', 'aviat', 'Google Drive', 'dl4us',
                            'prj')
    crawl(dir_path, 'eastern', 'singapore',
          'urban street -fashion -food -painting -art -graffiti', 300)
    crawl(dir_path, 'eastern', 'singapore',
          'urban road -fashion -food -painting -art -graffiti', 300)

    # western_city_and_keyword = {
    #     'nyc': (
    #         ('street -fashion -food -painting -art -graffiti', 600),
    #         ('urban street -fashion -food -painting -art -graffiti', 300),
    #         ('urban road -painting -traffic -map', 300),
    #     ),
    #     'vancouver': (
    #         ('street -fashion -food -painting -art -graffiti', 600),
    #         ('urban street -fashion -food -painting -art -graffiti', 300),
    #         ('urban road -painting -traffic -map', 300),
    #     ),
    #     'london': (
    #         ('street -fashion -food -painting -art -graffiti', 600),
    #         ('urban street -fashion -food -painting -art -graffiti', 300),
    #         ('urban road -painting -traffic -map', 300),
    #     ),
    #     'paris': (
    #         ('street -fashion -food -painting -art -graffiti', 300),
    #         ('rue image', 300),
    #         ('avenue image', 300),
    #         ('boulevard image', 300),
    #         ('sidewalk', 300),
    #     ),
    #     'moscow': (('street -fashion -food -painting -art -graffiti', 300), ),
    #     'Москва': (
    #         ('улица', 600),  # street
    #         ('тротуар', 300),  # sidewalk
    #     ),
    # }

    # eastern_city_and_keyword = {
    #     'tokyo': (('street -fashion -food -painting -art -graffiti', 600), ),
    #     'kyoto': (('street -fashion -food -painting -art -graffiti', 600), ),
    #     'beijing': (('street -fashion -food -painting -art -graffiti', 600), ),
    #     'singapore': (('street -fashion -food -painting -art -graffiti',
    #                    600), ),
    #     'seoul': (('street -fashion -food -painting -art -graffiti', 600), ),
    #     '東京': (
    #         ('路地', 300),
    #         ('道', 300),
    #         ('歩道', 300),
    #     ),
    #     '京都': (
    #         ('路地', 300),
    #         ('道', 300),
    #         ('歩道', 300),
    #     ),
    #     '北京': (
    #         ('街 -美食)', 300),  # street
    #         ('路 -美食)', 300),  # road
    #         ('人行道', 300),     # sidewalk
    #     ),
    #     '新加坡': (
    #         ('街 -美食)', 300),  # street
    #         ('路 -美食)', 300),  # road
    #         ('人行道', 300),     # sidewalk
    #     ),
    #     '서울': (
    #         ('거리', 300),      # street
    #         ('도로', 300),      # road
    #         ('보도', 300),      # sidewalk
    #     ),
    # }

    # supers = ('western', 'eastern')
    # subs = (western_city_and_keyword, eastern_city_and_keyword)
    # for super_concept, city_and_keyword in zip(supers, subs):
    #     for city_name, keyword_nums in city_and_keyword.items():
    #         for keyword, num_per_keyword in keyword_nums:
    #             crawl(
    #                 dir_path,
    #                 super_concept,
    #                 city_name,
    #                 keyword,
    #                 num_per_keyword=num_per_keyword,
    #             )
