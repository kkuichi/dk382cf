import numpy as np
import cv2

class Detector(object):

    def determine_building_orientations(self, rects):
        """
        Funkcia slúži na určenie orientácie objektov na základe rotovaných
        ohraničujúcich obdĺžnikov. Pre každý obdĺžnik získavame jeho rozmery
        a uhol natočenia. Ak je šírka menšia ako výška, uhol upravujeme o
        90 stupňov. Následne hodnotu normalizujeme a korigujeme o odchýlku 
        13 stupňov smerom na východ. Výsledkom je zoznam orientácií
        jednotlivých objektov.
        """

        orientations = []

        for rect in rects:
            (width, height), angle = rect[1], rect[2]

            if width < height:
                angle = 90 + angle 

            if angle < -45:
                orientation = angle + 360
            else:
                orientation = angle

            orientation = (orientation + 13) % 360

            orientations.append(orientation)

        return orientations


    def compute_difference(self, house, house_height):
        """
        Funkcia slúži na výpočet výškového rozdielu medzi objektom a jeho okolím.
        Pre každý objekt pracujeme s jeho binárnou maskou a príslušnými výškovými
        dátami z digitálneho výškového modelu. Najprv kontrolujeme, či vstupné dáta
        nie sú prázdne a či majú zhodné rozmery, pričom v prípade potreby ich
        upravujeme na spoločnú veľkosť. Následne vypočítavame priemernú výšku
        vnútri objektu a mimo objektu a ich rozdiel ukladáme ako výsledný príznak.
        Výstupom funkcie výškový rozdiel týchto dvoch priemerných výšok.
        """

        difference = []

        for i in range(len(house)):
            if house[i] is None or house[i].size == 0:
                difference.append(0)
                continue
            house_shape = house[i].shape
            house_height_shape = house_height[i].shape

            if house_shape != house_height_shape:
                min_shape = (min(house_shape[0], house_height_shape[0]), min(house_shape[1], house_height_shape[1]))
                house[i] = house[i][:min_shape[0], :min_shape[1]]
                house_height[i] = house_height[i][:min_shape[0], :min_shape[1]]

            avrg = np.average(house_height[i][house[i] != 0])
            avrg2 = np.average(house_height[i][house[i] == 0])
            difference.append(avrg - avrg2)

        return difference

    def avrg_h(self, house, house_height):
        """
        Funkcia slúži na výpočet priemernej výšky objektu na základe jeho masky
        a príslušných výškových dát. Z výškového modelu vyberáme iba tie hodnoty,
        ktoré patria objektu (t. j. kde maska nadobúda nenulové hodnoty), a z nich
        vypočítame priemer.

        Výstupom funkcie je jedna číselná hodnota reprezentujúca priemernú výšku
        daného objektu.
        """
        avrg = np.average(house_height[house != 0])
        return float(avrg)

    def compute_edge_heights(self, house_mask, height_data):
        """
        Funkcia slúži na výpočet priemernej výšky na okrajoch objektu.
        Najprv prevádzame masku objektu do 8-bitového formátu a pomocou
        detekcie hrán (Canny) získavame pixely reprezentujúce okraj objektu.
        Následne z výškových dát vyberáme hodnoty zodpovedajúce týmto hranám
        a vypočítavame ich priemer.

        Výstupom funkcie je priemerná výška okrajov objektu. V prípade, že
        sa nepodarí získať žiadne hrany, funkcia vracia hodnotu None.
        """
        house_mask = house_mask.astype(np.uint8) * 255
        edges = cv2.Canny(house_mask, 50, 150)

        edge_heights = height_data[edges == 255]

        if edge_heights.size > 0:
            mean_edge_height = np.mean(edge_heights)
            return mean_edge_height
        else:
            return None

    def detect_juego_pairs(self, houses, house_cord, difference, orientation):
        """
        Funkcia slúži na detekciu dvojíc objektov, ktoré spolu tvoria štruktúru typu „juego“.
        Prechádzame všetky detegované objekty a hľadáme také dvojice, ktoré spĺňajú
        definované geometrické a výškové podmienky.

        Pre každý objekt najprv overujeme, či jeho rozmery a výškový rozdiel zodpovedajú
        charakteristickým vlastnostiam jednej časti „juego“. Následne hľadáme druhý objekt,
        ktorý má podobné rozmery, podobnú výšku a orientáciu.

        Pri párovaní kontrolujeme rozdiel v orientácii, vzdialenosť medzi objektmi
        (vrátane povolenia mierneho prekryvu) a ich vzájomnú polohu v osi Y.
        Cieľom je nájsť dve približne rovnobežné a blízko seba umiestnené štruktúry.

        Pre každú nájdenú dvojicu vypočítavame stred spoločného objektu a ukladáme
        informácie o pôvodných indexoch a polohe. Každý objekt môže byť použitý
        maximálne raz.

        Výstupom funkcie je zoznam detegovaných dvojíc typu „juego“.
        """
        juego_pairs = []
        used = set()

        for a in range(len(houses)):
            i = houses[a][7]
            x1, y1, w1, h1 = house_cord[i]
            diff1 = difference[i]

            short1 = min(w1, h1)
            long1 = max(w1, h1)

            if not (17 <= short1 <= 22 and 35 <= long1 <= 45 and diff1 >= 1.5):
                continue

            for b in range(a + 1, len(houses)):
                j = houses[b][7]
                if i in used or j in used:
                    continue

                x2, y2, w2, h2 = house_cord[j]
                diff2 = difference[j]

                short2 = min(w2, h2)
                long2 = max(w2, h2)

                if not (17 <= short2 <= 22 and 35 <= long2 <= 45 and diff2 >= 1.5):
                    continue

                if abs(h1 - h2) > 5:
                    continue
                if abs(w1 - w2) > 5:
                    continue

                angle_i = orientation[i]
                angle_j = orientation[j]

                angle_diff = abs(angle_i - angle_j)
                angle_diff = min(angle_diff, 360 - angle_diff)

                if angle_diff > 15:
                    continue

                left1, right1 = x1, x1 + w1
                left2, right2 = x2, x2 + w2

                if right1 <= left2:
                    gap = left2 - right1
                elif right2 <= left1:
                    gap = left1 - right2
                else:
                    gap = -1

                if gap > 10:
                    continue

                if abs(y1 - y2) > 10:
                    continue

                cx = (x1 + w1 / 2 + x2 + w2 / 2) / 2
                cy = (y1 + h1 / 2 + y2 + h2 / 2) / 2

                juego_pairs.append({
                    "orig_i": i,
                    "orig_j": j,
                    "center_x": cx,
                    "center_y": cy
                })

                used.add(i)
                used.add(j)
                break

        return juego_pairs


    def find_buildings(self, img, house, angles, house_size, house_cord, difference, house_height):
        """
        Funkcia predstavuje hlavný klasifikačný mechanizmus systému na detekciu a
        zaradenie objektov do jednotlivých tried. Na vstupe prijíma zoznam masiek
        objektov, ich orientáciu, rozmery, súradnice, výškové rozdiely a výškové
        dáta odvodené z digitálneho výškového modelu.

        V prvej fáze pre každý objekt vypočítavame jeho základné geometrické a
        výškové príznaky. Určujeme najmä polohu stredu objektu, priemernú výšku,
        plochu, pomer strán, mieru zaplnenia ohraničujúceho obdĺžnika, kompaktnosť
        tvaru, približný počet vrcholov, tvarovú pravidelnosť a variabilitu výšky.
        Zároveň vyhodnocujeme aj rozdiel medzi priemernou výškou stredu a okrajov
        objektu.

        Následne aplikujeme počiatočný filter, ktorým odstraňujeme objekty, ktoré
        nespĺňajú minimálne požiadavky na plochu alebo nevykazujú dostatočne
        „building-like“ vlastnosti. Tým ponechávame len také objekty, ktoré majú
        zmysel ďalej klasifikovať.

        V ďalšej časti prebieha samotná klasifikácia pomocou súboru pravidiel.
        Objekty postupne porovnávame s podmienkami pre jednotlivé kategórie, ako sú
        menšie kompaktné stavby, domy, dvojdomy, platformy, ruiny, palácové objekty,
        pyramídy, clustre objektov a neznáme kategórie. Zaradenie je založené na
        kombinácii rozmerov, tvarových vlastností a výškových príznakov.

        Ak je k dispozícii vstupný obraz, funkcia priebežne vykresľuje
        ohraničujúce obdĺžniky objektov podľa výslednej triedy. Pre každý úspešne
        klasifikovaný objekt následne ukladá jeho stred, triedu, orientáciu,
        priemernú výšku, rozmery a ďalšie pomocné informácie.

        V druhej fáze funkcia dodatočne vyhľadáva dvojice objektov tvoriace štruktúru
        typu „juego“. Ak sa takáto dvojica nájde, obom objektom sa zmení trieda na
        `12_juego` a pri vizualizácii sa zvýraznia príslušnou farbou.

        Výstupom funkcie je zoznam finálne klasifikovaných objektov spolu s ich
        príznakmi a zaradením do tried.
        """
        houses = []

        #1. prvá fáza - klasifikácia jednotlivých objektov
        for i in range(len(house_size)):
            w, h = house_size[i]
            short_side = min(w, h)
            long_side = max(w, h)
            diff = float(difference[i]) if i < len(difference) else 0.0

            x, y, bw, bh = house_cord[i]

            mask = house[i].astype(np.uint8)

            if mask is None or mask.size == 0:
                continue

            #centroid
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx_local = int(M["m10"] / M["m00"])
                cy_local = int(M["m01"] / M["m00"])
            else:
                cx_local = bw // 2
                cy_local = bh // 2

            center_x = x + cx_local
            center_y = y + cy_local

            #priemerná výška
            try:
                avg_h = self.avrg_h(house[i], house_height[i])
            except Exception:
                avg_h = 0.0

            #základné príznaky
            obj_area = float(np.sum(mask != 0))
            bbox_area = float(mask.shape[0] * mask.shape[1]) if mask.size > 0 else 1.0
            extent = obj_area / bbox_area if bbox_area > 0 else 0.0
            aspect_ratio = (long_side / short_side) if short_side > 0 else 999.0

            mask255 = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            solidity = 0.0
            approx_vertices = 0
            rectangularity = 0.0

            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                cnt_area = cv2.contourArea(cnt)

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = cnt_area / hull_area if hull_area > 0 else 0.0

                perimeter = cv2.arcLength(cnt, True)
                epsilon = 0.02 * perimeter if perimeter > 0 else 1.0
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                approx_vertices = len(approx)

                rect = cv2.minAreaRect(cnt)
                (cx_rect, cy_rect), (rw, rh), angle = rect
                rw, rh = rect[1]
                rect_area = rw * rh
                rectangularity = cnt_area / rect_area if rect_area > 0 else 0.0

            #výškové príznaky vo vnútri objektu
            height_std = 0.0
            center_minus_edge = 0.0
            try:
                obj_heights = house_height[i][mask != 0]
                if obj_heights.size > 0:
                    height_std = float(np.std(obj_heights))

                    edge_h = self.compute_edge_heights(mask, house_height[i])
                    if edge_h is not None:
                        center_minus_edge = float(avg_h - edge_h)
            except Exception:
                pass

            if(obj_area < 50):
                continue

            #building-like filter
            building_like = (
                (obj_area >= 50 and solidity >= 0.60 and diff >= 0.08) or
                (obj_area >= 50 and extent >= 0.30 and rectangularity >= 0.45) or
                (obj_area >= 120 and diff >= 0.15) or
                (obj_area >= 180 and solidity >= 0.45) or
                (obj_area >= 55 and solidity >= 0.50 and extent >= 0.25 and diff >= 0.00)
            )

            if not building_like:
                print(
                    f"Vyradený objekt {i}: "
                    f"area={obj_area:.1f}, solidity={solidity:.2f}, extent={extent:.2f}, "
                    f"rect={rectangularity:.2f}, diff={diff:.2f}, w={w}, h={h}"
                )
                continue

            label = None
            color = (255, 255, 255)

            #1. malé kompaktné objekty
            if label is None and short_side <= 12 and long_side <= 18:
                #malé ruiny
                if diff < 0.20 and solidity >= 0.72 and extent >= 0.38:
                    label = "ruin_small"
                    color = (140, 220, 140)

                elif (
                    6 <= short_side <= 12 and
                    8 <= long_side <= 18 and
                    diff >= 1.00 and
                    extent >= 0.42 and
                    solidity >= 0.80
                ):
                    label = "03_palace"
                    color = (0, 0, 255)

                elif (
                    8 <= short_side <= 10 and
                    16 <= long_side <= 19 and
                    1.8 <= aspect_ratio <= 2.1 and
                    0.30 <= diff <= 0.80 and
                    solidity >= 0.90 and
                    rectangularity >= 0.80 and
                    height_std <= 0.40 and
                    center_minus_edge <= 0.50
                ):
                    label = "15_platform_small"
                    color = (255, 255, 0)
                    
                elif (
                    3 <= short_side <= 9 and
                    6 <= long_side <= 18 and
                    0.20 <= diff < 1.00 and
                    extent >= 0.35 and
                    solidity >= 0.65
                ):
                    label = "01_house"
                    color = (0, 255, 0)

                elif (
                    20 < obj_area < 65 and
                    3 <= short_side <= 10 and
                    6 <= long_side <= 16 and
                    0.20 <= diff < 1.00 and
                    solidity >= 0.72 and
                    rectangularity >= 0.50
                ):
                    label = "01_house"
                    color = (0, 255, 0)
        
            #2A. platformy
            #najprv malé, potom stredné, potom veľké kompaktné
            if label is None:
                if (
                    12 <= short_side <= 20 and
                    16 <= long_side <= 24 and
                    aspect_ratio <= 1.40 and
                    0.60 <= diff <= 1.20 and
                    solidity >= 0.90 and
                    rectangularity >= 0.75 and
                    height_std <= 0.65 and
                    center_minus_edge <= 0.85
                ):
                    label = "16_platform_large"
                    color = (0, 255, 255)
                #malé platformy
                if (
                    4 <= short_side <= 9 and
                    10 <= long_side <= 22 and
                    0.10 <= diff <= 0.90 and
                    solidity >= 0.72 and
                    rectangularity >= 0.42 and
                    height_std <= 0.55 and
                    center_minus_edge <= 0.22
                ):
                    label = "15_platform_small"
                    color = (255, 255, 0)

                #stredné platformy
                elif (
                    7 <= short_side <= 20 and
                    10 <= long_side <= 28 and
                    0.15 <= diff <= 1.30 and
                    solidity >= 0.68 and
                    rectangularity >= 0.42 and
                    height_std <= 0.65 and
                    center_minus_edge <= 0.40
                ):
                    label = "16_platform_large"
                    color = (0, 255, 255)

                #väčšie kompaktné platformy
                elif (
                    short_side >= 18 and
                    long_side >= 20 and
                    aspect_ratio <= 1.5 and
                    0.30 <= diff <= 1.20 and
                    center_minus_edge <= 0.40 and
                    height_std <= 1.20 and
                    solidity >= 0.75 and
                    rectangularity >= 0.55
                ):
                    label = "16_platform_large"
                    color = (0, 255, 255)

            #2B. podlhovasté objekty
            if label is None and aspect_ratio >= 1.35:
                #ruinové podlhovasté objekty
                if diff < 0.20 and obj_area >= 50 and solidity >= 0.58 and extent >= 0.28:
                    label = "ruin_elongated"
                    color = (120, 200, 120)

                #menší double house
                elif (
                    8 <= short_side <= 14 and
                    16 <= long_side <= 24 and
                    1.40 <= aspect_ratio <= 2.20 and
                    solidity >= 0.65
                ):
                    label = "02_double_house"
                    color = (0, 180, 0)

                #väčší double house
                elif (
                    8 <= short_side <= 20 and
                    25 <= long_side <= 50 and
                    aspect_ratio >= 2.0 and
                    solidity >= 0.70
                ):
                    label = "02_double_house"
                    color = (0, 180, 0)

                #veľmi pravidelný úzky double house
                elif (
                    8 <= short_side <= 12 and
                    24 <= long_side <= 30 and
                    solidity >= 0.88
                ):
                    label = "02_double_house"
                    color = (0, 180, 0)

            #3. veľké / pravidelné objekty
            #poradie: ruina -> pyramída -> palace -> palace_large -> clustre
            if label is None and obj_area >= 80:
                #veľká ruina
                if diff < 0.25 and solidity >= 0.62 and rectangularity >= 0.42:
                    label = "ruin_large_rectangular"
                    color = (150, 200, 150)

                #pyramída
                elif (
                    short_side >= 20 and
                    long_side >= 20 and
                    aspect_ratio <= 1.45 and
                    diff >= 3.0 and
                    center_minus_edge >= 0.8 and
                    solidity >= 0.72 and
                    rectangularity >= 0.55 and
                    approx_vertices <= 8
                ):
                    label = "11_temple_pyramida"
                    color = (255, 0, 0)

                #stredne veľký palace
                elif (
                    8 <= short_side <= 18 and
                    12 <= long_side <= 28 and
                    0.35 <= diff < 3.0 and
                    solidity >= 0.85 and
                    rectangularity >= 0.50
                ):
                    label = "04_palace"
                    color = (0, 0, 200)
                
                # kompaktný double house
                elif (
                    16 <= short_side <= 20 and
                    18 <= long_side <= 24 and
                    aspect_ratio <= 1.20 and
                    0.70 <= diff <= 1.30 and
                    0.75 <= solidity <= 0.88 and
                    0.50 <= extent <= 0.65 and
                    rectangularity <= 0.68 and
                    approx_vertices >= 7
                ):
                    label = "02_double_house"
                    color = (0, 180, 0)

                #veľký palace objekt
                elif (
                    obj_area > 200 and
                    18 <= short_side <= 30 and
                    20 <= long_side <= 60 and
                    0.70 <= diff <= 3.50 and
                    solidity >= 0.72 and extent >= 0.50
                ):
                    label = "04_palace_large"
                    color = (0, 0, 150)

            #4. nepravidelné clustre
            if label is None:
                if (
                    obj_area > 400 and
                    aspect_ratio < 2.0 and
                    solidity < 0.75
                ):
                    label = "cluster_houses"
                    color = (180, 180, 180)

                elif (
                    obj_area > 200 and
                    solidity < 0.70 and
                    extent < 0.50
                ):
                    label = "cluster_houses"
                    color = (180, 180, 180)

            #5. fallback unknown kategórie
            if label is None:
                if diff < 0.15:
                    if aspect_ratio >= 1.6:
                        label = "unknown_elongated_low"
                    elif obj_area >= 150:
                        label = "unknown_large_low"
                    else:
                        label = "unknown_small_low"
                    color = (120, 120, 120)

                else:
                    if aspect_ratio >= 1.8:
                        label = "unknown_elongated_building"
                    elif obj_area >= 180:
                        label = "unknown_large_building"
                    else:
                        label = "unknown_small_building"
                    color = (80, 80, 80)

            #debug box
            if img is not None:
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 1)

            houses.append([
                center_x,
                center_y,
                label,
                angle,
                avg_h,
                rw,
                rh,
                i,
                cnt
            ])

        #druha faza: urcenie juego
        juego_pairs = self.detect_juego_pairs(houses, house_cord, difference, angles)
        if juego_pairs:
            for pair in juego_pairs:
                i = pair["orig_i"]
                j = pair["orig_j"]

                for hinfo in houses:
                    if hinfo[7] == i or hinfo[7] == j:
                        hinfo[2] = "12_juego"

                x1, y1, w1, h1 = house_cord[i]
                x2, y2, w2, h2 = house_cord[j]

                if img is not None:
                    cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 255), 1)
                    cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 255), 1)

        return houses  

   

