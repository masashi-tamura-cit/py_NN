import glob
from os.path import join
from consts import OUTPUT_BASE_DIR


def scraping_data(dir_name: str) -> None:
    """
    :param dir_name:
    :return:
    """
    base_dir_list = glob.glob(join(dir_name, "*"))
    for base_dir in base_dir_list:
        dir_list = glob.glob(join(base_dir, "*"))
        for file_dir in dir_list:
            file_list = glob.glob(join(file_dir, "*"))
            datas = []
            for file in file_list:
                print(f"target_file_path: {file}, scraping start\n")
                fp = open(file, mode="r")
                rows = (fp.readlines())
                length = str(len(rows))
                datas.append(length + "," + rows[-1])
                fp.close()
            header = "epochs,train_accuracy,train_error,accuracy,error,L1_norm,L2_norm,node_amount,epoch_time,total_time,weight_active_ratio\n"
            path = join(file_dir, "data.csv")
            with open(path, mode="w", newline="") as f:
                f.write(header)
                for data in datas:
                    f.write(data)
                    f.write("\n")
            print(f"make_file, {path}\n")


if __name__ == "__main__":
    print("start_scraping\n")
    scraping_data(OUTPUT_BASE_DIR)
