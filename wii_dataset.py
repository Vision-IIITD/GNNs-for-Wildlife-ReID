import pandas as pd
from wildlife_datasets import datasets
import os
import json

class WII(datasets.DatasetFactory):
    def create_catalogue(self) -> pd.DataFrame:
        path_images = os.path.join("wii.coco", "images")
        path_json = os.path.join("wii.coco", "instances_all2017.json")
    
        with open(os.path.join(self.root, path_json), "r") as file:
            data = json.load(file)

        create_dict = lambda i: {'identity': i['identity'], 'bbox': i['bbox'], 'image_id': i['image_id'], 'category_id': i['category_id'], 'split': i['split']}
        df_anns = pd.DataFrame([create_dict(i) for i in data['annotations']])
        create_dict = lambda i: {'file_name': i['file_name'], 'id': i['id']}
        df_images = pd.DataFrame([create_dict(i) for i in data['images'] if os.path.exists(os.path.join(self.root, path_images, i['file_name']))])

        df = df_anns.merge(df_images, left_on='image_id', right_on='id', how='left')
        df = df.drop(columns=['id'])
        df = df.rename(columns={'file_name': 'path'})
        df = df.dropna(subset=['path'])

        # drop rows with missing images

        df['path'] = df['path'].apply(lambda x: os.path.join(path_images, str(x)))

        df = df[['image_id', 'path', 'identity', 'split', 'bbox', 'category_id']]

        gallery_metdata_train = os.path.join("wii.coco", "gallery_metadata_train.json")
        with open(os.path.join(self.root, gallery_metdata_train), "r") as file:
            gallery_json_train = json.load(file)

        gallery_metdata_test = os.path.join("wii.coco", "gallery_metadata_test.json")
        with open(os.path.join(self.root, gallery_metdata_test), "r") as file:
            gallery_json_test = json.load(file)

        df = self._add_flank_split(df, gallery_json_train)
        df = self._add_flank_split(df, gallery_json_test)

        return self.finalize_catalogue(df)

    def _add_flank_split(self, df: pd.DataFrame, gallery: dict) -> pd.DataFrame:
        for tiger_item in gallery:
            for gal_lf in tiger_item['gallery']["left_flank"]:
                img_path = os.path.join("wii.coco", "images", gal_lf)
                df.loc[df['path'] == img_path, 'gallery_split'] = 'gallery'
                df.loc[df['path'] == img_path, 'flank'] = 'left'

            for gal_rf in tiger_item['gallery']["right_flank"]:
                img_path = os.path.join("wii.coco", "images", gal_rf)
                df.loc[df['path'] == img_path, 'gallery_split'] = 'gallery'
                df.loc[df['path'] == img_path, 'flank'] = 'right'

            for query_lf in tiger_item['query']["left_flank"]:
                img_path = os.path.join("wii.coco", "images", query_lf)
                df.loc[df['path'] == img_path, 'gallery_split'] = 'query'
                df.loc[df['path'] == img_path, 'flank'] = 'left'

            for query_rf in tiger_item['query']["right_flank"]:
                img_path = os.path.join("wii.coco", "images", query_rf)
                df.loc[df['path'] == img_path, 'gallery_split'] = 'query'
                df.loc[df['path'] == img_path, 'flank'] = 'right'

        return df
    

if __name__ == "__main__":
    df = WII("datasets/WII").create_catalogue()