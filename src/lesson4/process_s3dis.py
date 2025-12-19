import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

def process_s3dis_area(area_path, class_mapping):
 
    if not os.path.isdir(area_path):
        print(f"Директория не найдена: {area_path}")
        return None

    area_data = []
    room_list = [d for d in os.listdir(area_path) if os.path.isdir(os.path.join(area_path, d))]

    for room_name in room_list:
        room_path = os.path.join(area_path, room_name, 'Annotations')

        if not os.path.isdir(room_path):
            continue

        for annotation_file in os.listdir(room_path):
            if annotation_file.endswith('.txt'):
                class_name = annotation_file.split('_')[0]
                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)
                label = class_mapping[class_name]

                file_path = os.path.join(room_path, annotation_file)
                try:
                    point_cloud = np.loadtxt(file_path, delimiter=' ')
                    labels_column = np.full((point_cloud.shape[0], 1), label)
                    room_class_data = np.hstack((point_cloud, labels_column))
                    area_data.append(room_class_data)
                except Exception as e:
                    print(f"Ошибка при чтении файла {file_path}: {e}")

    if not area_data:
        return None

    full_area_data = np.vstack(area_data)
    coords_rgb = full_area_data[:, :6]
    labels = full_area_data[:, 6]

    coords_rgb[:, 3:] /= 255.0

    coords_mean = np.mean(coords_rgb[:, :3], axis=0)
    coords_rgb[:, :3] -= coords_mean
    max_abs_val = np.max(np.abs(coords_rgb[:, :3]))
    coords_rgb[:, :3] /= max_abs_val
    
    processed_data = np.hstack((coords_rgb, labels[:, np.newaxis]))
    return processed_data

def visualize_label_distribution(labels, output_filename='label_distribution.png'):
 
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    plt.bar(unique_labels, counts, tick_label=[str(int(l)) for l in unique_labels])
    plt.xlabel('Метка класса')
    plt.ylabel('Количество точек')
    plt.title('Распределение меток классов в датасете')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"\nГистограмма распределения меток сохранена в файл: {output_filename}")


def main(dataset_path):
    """
    Главная функция для обработки датасета S3DIS.
    """
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        print(f"Ошибка: Путь к датасету '{dataset_path}' не существует.")
        return

    all_areas_data = []
    class_mapping = {}


    # Обрабатываем только 'Area_1' для экономии времени и места.

    # area_folders = sorted([d for d in os.listdir(dataset_path) if d.startswith('Area_')])
    area_folders = ['Area_1']
    print(f"ВНИМАНИЕ: Для оптимизации будет обработана только папка: {area_folders}")
    
    if not area_folders or not os.path.exists(os.path.join(dataset_path, area_folders[0])):
        print(f"Ошибка: В директории '{dataset_path}' не найдена папка 'Area_1'.")
        return

    for area_name in area_folders:
        print(f"Обработка {area_name}...")
        area_path = os.path.join(dataset_path, area_name)
        processed_area_data = process_s3dis_area(area_path, class_mapping)
        if processed_area_data is not None:
            all_areas_data.append(processed_area_data)

    if not all_areas_data:
        print("Не было найдено данных для обработки.")
        return
        
    print("\nОбъединение данных...")
    final_dataset = np.vstack(all_areas_data)

    print("\nСохранение данных...")

    np.save('s3dis_dataset.npy', final_dataset)
    print("Данные сохранены в s3dis_dataset.npy")
    

    print("\nПервые 5 строк итогового массива:")
    print(final_dataset[:5, :])
    
    with open('s3dis_head_output.txt', 'w') as f:
        f.write("Первые 5 строк итогового массива:\n")
        f.write(str(final_dataset[:5, :]))

    # 5. Визуализация распределения меток
    labels = final_dataset[:, -1]
    visualize_label_distribution(labels)

    print("\nСкрипт успешно завершил работу!")
    print("Созданы следующие файлы для отчета:")
    print("- s3dis_dataset.npy (основной файл с данными, ~2-3 ГБ)")
    print("- s3dis_head_output.txt (скриншот первых 5 строк)")
    print("- label_distribution.png (диаграмма)")

if __name__ == '__main__':
    dataset_path = 'Stanford3dDataset_v1.2_Aligned_Version'
    main(dataset_path)
