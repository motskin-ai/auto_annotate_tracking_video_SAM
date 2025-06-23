import cv2
import json
import numpy as np
import os
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import shutil
import subprocess
import torch


video_file = "/mnt/e/!MotskinAI/tracking/02_sam_autoannotation_tracking_in_video/videos/f-16.mp4"    # местоположение исходного видео 
temp_video_dir = f"/mnt/r/temp_sam"                             # временная папка для сохранения раскадровки видео

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"             # используемые веса для модели
model_cfg = "configs/sam2.1_hiera_l.yaml"                                 # конфигурация модели

width = 1800                            # размер экрана для отображения. можно было и автоматически рассчитать...
height = 1000
RADIUS_OF_SEARCH_AREA = 11               # Радиус окрестности поиска
should_draw_bbox = True

file_name_only = os.path.splitext(os.path.basename(video_file))[0]      # разметка будет иметь имя файла такое же как и имя видео, только json
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()    # используем bfloat16 для вычисления

if torch.cuda.get_device_properties(0).major >= 8:
    # Если архитектура GPU Ampere и выше (т.е. RTX30, и выше) то можно разрешить использования тензорных ядер tfloat32,
    # которые должны улучшить производительность (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True    # разрешаем операцию matmul на тензорных ядрах
    torch.backends.cudnn.allow_tf32 = True          # разрежаем выполнять свёртки на тензорных ядрах

frame_prompt_point_dict = dict()    # хранит номер фрейма и его точки
current_index = 0                   # индекс текущего отображаемого кадра
old_index = -1                      # индекс предыдущего отображаемого кадра (чтобы не перерисовывать неизменяемое изображение)
prompts = dict()
video_segments = dict()             # здесь будут храниться результаты сегментации с помощью SAM для всех кадров

colors = {   # словарь для цвета маски в зависимости от объекта (В начальной версии используем только один цвет)
    0: np.concatenate([np.array([255, 0, 0]), np.array([0.7])], axis=0)
}                     


def draw_points(img, points, radius=RADIUS_OF_SEARCH_AREA):
    """ Отрисовываем точки-промты (зелёные точки - это позитивные, а красные - негативные)
        :param img: Исходное изображение
        :param points: Коллекция точек-промтов
        :param radius: радиус окружности 
    """
    for (pos, label) in points.items():
        point_color = (0, 255, 0) if label == 1 else (0, 0, 255)      # позитивные точки-запросы вывожу зелёным цветом, а негативные - красным
        cv2.circle(img, pos, radius, point_color, -1)


def refresh_image():
    global current_index, total
    global frame_names, frame_prompt_point_dict
    global should_draw_bbox
    global frame_width, frame_height
    global width, height

    # изображения загружены и в переменную inference_state для SAM, но там они уже масштабированы, поэтому проще работать с оригиналом, будем непосредственно читать из файла
    img = cv2.imread(os.path.join(temp_video_dir, frame_names[current_index]))      # читаем изображение из файла
    cv2.putText(img, f"{current_index + 1}/{total}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)     # выводим номер текущего кадра и сколько всех кадров

    img = process_image(img)
    
    if current_index in frame_prompt_point_dict:            # если для этого изображения есть точки-запросы, 
        points = frame_prompt_point_dict[current_index]         # то собирём их все и далее
        draw_points(img, points)                                # отображим на изображении
    
    if should_draw_bbox and current_index in video_segments:
        mask_dict = video_segments[current_index]       # получаем маски для каждого объекта
        mask = mask_dict[0]                         # но пока поддерживается в коде только для одной маски
        bbox = find_box_in_mask(mask)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2,y2), (0, 255, 255), 2)

    cv2.imshow("cam_wind", cv2.resize(img, (width, height)))    # здесь же и отображу, предварительно выполнив масштабирование до заданных размеров


# Обработчик событий мыши
def mouse_callback(event, x, y, flags, param):
    global frame_width, frame_height
    global width, height
    global frame_prompt_point_dict
    global current_index

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:    # левая и правая кнопка мыши ставит позитивную или негативную точку-запрос

        points = frame_prompt_point_dict.get(current_index, dict())         # нам нужно знать текущие установленные точки, чтобы можно было её удалить

        x = int(x * frame_width / width)            # мы работаем на масштабированном окне, поэтому точку клика нужно преобразовать в реальный размер кадра
        y = int(y * frame_height / height)          

        # ищем имеющиеся точки в окрестностях клика, нужно, чтобы удалить её если она там есть
        founded_point = None
        for point in points.keys():
            x_t, y_t = point
            if x >= x_t - RADIUS_OF_SEARCH_AREA - 1 and x <= x_t + RADIUS_OF_SEARCH_AREA + 1 and y >= y_t - RADIUS_OF_SEARCH_AREA - 1 and y <= y_t + RADIUS_OF_SEARCH_AREA + 1:
                 founded_point = point      # если нашли точку поблизости текущего клика
                 break

        if event == cv2.EVENT_LBUTTONDOWN:
            point_value = 1                     # позитивная метка
        elif event == cv2.EVENT_RBUTTONDOWN:
            point_value = 0                     # негативная метка

        if founded_point is not None and points[founded_point] == point_value:      # если нашли существующую точку и метка совподает с меткой клика, то
            del points[founded_point]                                               # значит нужно удалить точку
        else:
            points[(x, y)] = point_value                                            # иначе добавить. Ключём в словаре является кортеж (x, y)

        if len(points) > 0:                                         # если есть хоть одна точка-запрос для текущего кадра, то
            frame_prompt_point_dict[current_index] = points         # фиксируем эти точки в глобальном словаре для текущего кадра
        else:
            del frame_prompt_point_dict[current_index]              # а если больше не осталось точек, то полностью удаляем информацию о точках для текущего кадра
        
        refresh_image()                          # в любом случае маска должна как-то измениться, а значит нужно выполнить предсказание для текущего кадра и обновить изображение


def process_image(img):
    """ отрисовываем маску сегментированного объекта на изображении
    """
    global frame_prompt_point_dict, current_index
    global prompts, inference_state

    object_id = 0               # Внимание, текущая версия работает только с одним объектом, но местами в коде есть задел на работу с множеством объектов

    promt_pos_lst = list()                  # хранит все позиции точек-запросов
    promt_labels_lst = list()               # хранит все метки точек-запросов
    if len(frame_prompt_point_dict) == 0:       # если ещё нет ни одной точки-запроса, то ничего не нужно делать
        return img                                  # возратим исходное изображение

    if current_index in frame_prompt_point_dict:
        for pos, label in frame_prompt_point_dict[current_index].items():   # но если в памяти уже есть точки-запросы, то проходимся по ним.
            promt_pos_lst.append(list(pos))                         # заполняем коллекции и для позиций точек-запросов и их метки
            promt_labels_lst.append(label)

        points = np.array(promt_pos_lst, dtype=np.float32)              # далее точки-запросы должны храниться в виде numpy массива с элементами типа float32
        labels = np.array(promt_labels_lst, np.int32)                       # а метки также в виде numpy массива, только тип int32
        prompts[object_id] = points, labels                             # Добавляем в словарь запросов эти точки для конкретного объекта (в текущей версии программы он только один)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(     # Добавляем эти точки предсказателю для определённого кадра. 
            inference_state=inference_state,
            frame_idx=current_index,
            obj_id=object_id,
            points=points,
            labels=labels,
        )

        mask_logits = [(out_mask_logits[i] > 0.0).cpu().numpy() for i in out_obj_ids]   # получаем маски которые смогли сегментировать 
        img = draw_masks(img, mask_logits, out_obj_ids)                                 # отрисовываем эти маски на изображении для найденных объектов
    else:
        if current_index in video_segments:
            img = draw_masks(img, video_segments[current_index], [0])

    return img              # И возвращаем полученное изображение с наложенной маской.


def draw_masks(img, mask_logits, obj_ids):
    """ Отрисовываем маски на изображении на уровне массивов
        :param img: Изображение, на котором нужно выполнить отрисовку
        :param mask_logits: маски для объектов, которые будет поочереди накладывать на изображение
        :param obj_ids: идентификаторы объектов
        :return: Изображение с наложенными масками
    """
    for i, out_obj_id in enumerate(obj_ids):    # проходим по каждому объекту
        h, w = mask_logits[i].shape[-2:]
        mask = mask_logits[i].reshape(h, w, 1) * colors[out_obj_id].reshape(1, 1, -1)       # трюк, чтобы получить маску с заданным цветом для определённого объекта

        foreground_colors = mask[:, :, :3]  # получаем основные цвета маски
        alpha_channel = mask[:, :, 3]       # получаем альфа канал для маски.
        alpha_mask = alpha_channel[:, :, np.newaxis]    # для следующей операции нужно альфаканалу добавить 3-е измерение, чтобы все матрицы были одной размерности

        img = img * (1 - alpha_mask) + foreground_colors * alpha_mask       # выполняем наложение маски. Изменение будут только для цветных пикселей маски

    return np.array(img, dtype=np.uint8)        # восстанавливаем привычный тип данных uint8 для изображений и возвращаем его с наложенной маской.


def propogate():
    """ С помощью SAM выполняем предскажание для всех кадров с учётом точек-запросов
    """
    # запускаем распространение по всему видео и собираем результаты в dict
    temp_video_segments = {}  # video_segments содержит результаты сегментации по каждому кадру
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        temp_video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return temp_video_segments      


def find_box_in_mask(mask: np.array):
    """ Ищет ограничивающую рамку для маски
        :param mask: маска, в которой нужно найти ограничивающую рамку
        :return: Кортеж из 4-х значений верней левой точки и правой нижней. (x_min, y_min, x_max, y_max). Или None, если маска не содержит выделенной области
    """
    coords = np.argwhere(mask[0])
    
    if coords.size == 0:    # Если маска не содержит выделенного сегмента
        return None  
    
    # Находим минимальные и максимальные координаты по осям
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return int(x_min), int(y_min), int(x_max), int(y_max)


def create_and_save_annotation(video_segments):
    """ Создаём и сохраняем файл разметки. Внимание, мы сохраняем меньше информации чем на самом деле у нас есть
        :param video_segments: результат сегментации по каждому кадру
    """
    annotation = dict()
    for frame_idx in video_segments:                # выбираем по одному результату сегментации, который для каждого кадра
        mask_dict = video_segments[frame_idx]       # получаем маски для каждого объекта

        mask = mask_dict[0]                         # но пока поддерживается в коде только для одной маски
        points = find_box_in_mask(mask)             # определяем точки ограничивающей рамки для этой маски
        annotation[frame_idx] = points              # и собираем их в словарь

    output_file_name = os.path.join(output_dir, f"{file_name_only}.json")
    with open(output_file_name, mode='w', encoding="utf-8") as f:
        json.dump(annotation, f)                                                # просто сохраняем словарь в json файл
    print(f"Файл успешно сохранён: {output_file_name}")
    

sam2_checkpoint = os.path.abspath(sam2_checkpoint)      # получаем абсолютный путь к весам модели
video_file = os.path.abspath(video_file)                # получаем абсолютный путь к файлу видео
temp_video_dir = os.path.abspath(temp_video_dir)        # получаем абсолютный путь к временной папке с кадрами
output_dir = os.path.dirname(video_file)                # папка, куда нужно будет поместить результат

if os.path.exists(temp_video_dir):          # для каждого запуска будем делать новую раскодровку, поэтому старые данные удаляем
    shutil.rmtree(temp_video_dir)
os.makedirs(temp_video_dir)

# Разбиваем видео на кадры с помощью ffmpeg
command = f"ffmpeg -i {video_file} -q:v 2 -start_number 0 {temp_video_dir}/%05d.jpg"
subprocess.run(command, shell=True)

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)   # создаём объект предсказатель SAM. Ему нужна конфигурация модели и предобученные веса
inference_state = predictor.init_state(video_path=temp_video_dir)    # подгружаем предсказателю исходные данные (а по факту все кадры видео)

# Получаем все название кадров в формате JPEG в указанном каталоге
frame_names = [
    p for p in os.listdir(temp_video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))              # выполняем их сортировку. Когда разбивали на кадры с помощью ffmpeg, то давали соответствующие имена кадрам, чтобы можно было восстановить исходный порядок кадров

frame_width, frame_height = Image.open(os.path.join(temp_video_dir, frame_names[0])).size       # получаем размер кадра, по первому кадру из видео

cv2.namedWindow("cam_wind", flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO |cv2.WINDOW_GUI_NORMAL)      # cv2.WINDOW_GUI_NORMAL - для того, чтобы не вываливолось меню при нажатии правой кнопкой мыши
cv2.setMouseCallback("cam_wind", mouse_callback)        # прописываем обработчик событий

total = len(frame_names)    # общее количество кадров

speed_change_frame = 1      # используется для динамического ускорения навигации

while True:
    if old_index != current_index:      # если мы с помощью навигации перешли на новый кадр
        old_index = current_index
        refresh_image()                 # то нужно перерисовать текущий кадр

    key = cv2.waitKey(1)
    if key == ord('q') or key==27:                                              # q - завершить программу
        break
    elif key == ord('d') and current_index > 0:                                 # d - предыдущий кадр
        step = min(speed_change_frame, current_index)       # определяем шаг прирощения (для навигации)
        current_index -= step
        speed_change_frame = 10
    elif (key == 32 or key == ord('f')) and current_index < total - 1:         # f или Пробел - следующий кадр
        step = min(speed_change_frame, total - current_index-1) # определяем шаг прирощения (для навигации)
        current_index += step
        speed_change_frame = 10
    elif key == ord('r'):                               # r - reset сбросить все точки на текущем кадре
        del frame_prompt_point_dict[current_index]
        refresh_image()
    elif key == ord('p'):                               # p - выполнить предсказание для всех кадров с помощью SAM
       video_segments = propogate()
       create_and_save_annotation(video_segments)
    elif key == ord('h'):                               # h - управляет нужно ли отображать ограничивающую рамку
        should_draw_bbox = not should_draw_bbox
        refresh_image()
    elif key == -1:
        speed_change_frame = 1


cv2.destroyAllWindows()

print("Программа завершена")