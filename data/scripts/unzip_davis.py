import zipfile
import os

def main():
    # 获取当前脚本的绝对路径，并推导出相关目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 路径构造
    zip_path = os.path.normpath(os.path.join(script_dir, '../raw/davis/DAVIS-2017-trainval-480p.zip'))
    extract_to = os.path.normpath(os.path.join(script_dir, '../raw/davis/'))
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} does not exist.")
        return

    print(f"Extracting:\n  Source: {zip_path}\n  Target: {extract_to}")
    
    # 解压文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    print("DAVIS-2017-trainval-480p.zip extraction complete.")

if __name__ == '__main__':
    main()
