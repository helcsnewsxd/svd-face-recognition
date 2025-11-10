import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


class SVDFaceRecognition:
    """
    Implementación del algoritmo SVD para reconocimiento facial siguiendo los pasos:
    1. Obtener conjunto de entrenamiento S con N imágenes faciales
    2. Calcular imagen media f̄ de S
    3. Formar matriz A con f̄ calculado
    4. Calcular SVD de A
    5. Calcular vector de coordenadas xi para cada individuo conocido
      a. Elegir umbral ε₁
      b. Elegir umbral ε₀
    6. Para nueva imagen: calcular coordenadas x, distancia εf, y clasificar
      a. Calcular coordenadas x
      b. Calcular distancia εf
      c. Clasificar la imagen
    """
    
    def __init__(self, epsilon_1: float = 0.3, epsilon_0: float = 0.2):
        """
        Inicializa el sistema SVD.
        
        Args:
            epsilon_1: Umbral ε₁ para distancia al espacio facial
            epsilon_0: Umbral ε₀ para distancia a cara conocida
        """
        self.epsilon_1 = epsilon_1
        self.epsilon_0 = epsilon_0
        
        # Variables del modelo entrenado
        self.mean_face = None  # f̄ (ecuación 18)
        self.U = None         # Matriz U de SVD
        self.singular_values = None  # Valores singulares
        self.V = None         # Matriz V de SVD
        self.training_coordinates = {}  # xi para cada persona
        self.person_labels = []   # Individuos conocidos
        self.is_trained = False
    
    def step_1_obtain_training_set(self, training_data: Dict[str, List[str]]) -> np.ndarray:
        """
        Paso 1: Obtener conjunto de entrenamiento S con N imágenes faciales de individuos conocidos.
        
        Args:
            training_data: Diccionario {persona: [rutas_imagenes]}
            
        Returns:
            Matriz S de tamaño M×N donde M=píxeles, N=imágenes
        """
        print("Paso 1: Obteniendo conjunto de entrenamiento...")
        
        all_faces = []
        self.person_labels = []
        
        for person, image_paths in training_data.items():
            print(f"  Procesando {len(image_paths)} imágenes de {person}")
            for image_path in image_paths:
                try:
                    # Cargar y preprocesar imagen
                    face_vector = self._preprocess_image(image_path)
                    all_faces.append(face_vector)
                    self.person_labels.append(person)
                except Exception as e:
                    print(f"    Error procesando {image_path}: {e}")
                    continue
        
        if len(all_faces) == 0:
            raise ValueError("No se pudieron cargar imágenes de entrenamiento")
        
        # Convertir a matriz S (M×N)
        S = np.array(all_faces).T
        print(f"  Conjunto de entrenamiento: {S.shape[0]} píxeles × {S.shape[1]} imágenes")
        
        return S
    
    def step_2_calculate_mean_face(self, S: np.ndarray) -> np.ndarray:
        """
        Paso 2: Calcular imagen media f̄ de S usando ecuación (18).
        
        Args:
            S: Matriz de entrenamiento M×N
            
        Returns:
            Imagen media f̄
        """
        print("Paso 2: Calculando imagen media...")
        
        # Ecuación (18): f̄ = (1/N) ∑fᵢ
        self.mean_face = np.mean(S, axis=1)
        
        print(f"  Imagen media calculada: {len(self.mean_face)} píxeles")
        return self.mean_face
    
    def step_3_form_matrix_A(self, S: np.ndarray) -> np.ndarray:
        """
        Paso 3: Formar matriz A en ecuación (20) con f̄ calculado.
        
        Args:
            S: Matriz de entrenamiento M×N
            
        Returns:
            Matriz A de diferencias M×N
        """
        print("Paso 3: Formando matriz A...")
        
        # Ecuación (19): aᵢ = fᵢ - f̄
        # Ecuación (20): A = [a₁, a₂, ..., aₙ]
        A = S - self.mean_face.reshape(-1, 1)
        
        print(f"  Matriz A formada: {A.shape[0]} píxeles × {A.shape[1]} imágenes")
        return A
    
    def step_4_calculate_svd(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Paso 4: Calcular SVD de A como se muestra en (1).
        
        Args:
            A: Matriz M×N
            
        Returns:
            Tupla (U, S, V) de la descomposición SVD
        """
        print("Paso 4: Calculando SVD...")
        
        # Ecuación (1): A = USV^T
        self.U, self.singular_values, Vt = np.linalg.svd(A, full_matrices=False)
        self.V = Vt.T
        
        print(f"  SVD calculado: {len(self.singular_values)} valores singulares")
        print(f"  U: {self.U.shape}, V: {self.V.shape}")
        
        return self.U, self.singular_values, self.V
    
    def step_5_calculate_training_coordinates(self, A: np.ndarray):
        """
        Paso 5: Para cada individuo conocido, calcular vector de coordenadas xᵢ
        a partir de (23).
        
        Args:
            A: Matriz de diferencias M×N
        """
        print("Paso 5: Calculando coordenadas de entrenamiento...")
        
        # Ecuación (23): xᵢ = [u₁, u₂, ..., uᵣ]^T (fᵢ - f̄)
        for i, person in enumerate(self.person_labels):
            if person not in self.training_coordinates:
                self.training_coordinates[person] = []
            
            # Calcular coordenadas para la imagen i
            face_vector = A[:, i]
            coordinates = self.U.T @ face_vector  # Ecuación (21)
            self.training_coordinates[person].append(coordinates)
        
        print(f"  Coordenadas calculadas para {len(self.training_coordinates)} personas")
    
    def step_5_a_b_choose_thresholds(self, epsilon_1: float = None, epsilon_0: float = None):
        """
        Paso 6: Elegir umbrales ε₁ y ε₀.
        
        Args:
            epsilon_1: Umbral ε₁ para distancia al espacio facial
            epsilon_0: Umbral ε₀ para distancia a cara conocida
        """
        print("Paso 5ab: Configurando umbrales...")
        
        if epsilon_1 is not None:
            self.epsilon_1 = epsilon_1
        if epsilon_0 is not None:
            self.epsilon_0 = epsilon_0
        
        print(f"  ε₁ (distancia al espacio facial): {self.epsilon_1}")
        print(f"  ε₀ (distancia a cara conocida): {self.epsilon_0}")
    
    def step_6a_calculate_coordinates(self, image_path: str) -> np.ndarray:
        """
        Paso 6a: Para nueva imagen f, calcular vector de coordenadas x
        a partir de ecuación (21).
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Coordenadas x en el espacio facial
        """
        # Preprocesar imagen
        face_vector = self._preprocess_image(image_path)
        
        # Ecuación (21): x = [u₁, u₂, ..., uᵣ]^T (f - f̄)
        diff = face_vector - self.mean_face
        coordinates = self.U.T @ diff
        
        return coordinates
    
    def step_6b_calculate_face_distance(self, image_path: str) -> float:
        """
        Paso 6b: Calcular distancia εf al espacio facial a partir de ecuación (25).
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Distancia al espacio facial
        """
        # Preprocesar imagen
        face_vector = self._preprocess_image(image_path)
        diff = face_vector - self.mean_face
        
        # Calcular coordenadas
        coordinates = self.U.T @ diff
        
        # Ecuación (24): fₚ = [u₁, u₂, ..., uᵣ] x
        face_projection = self.U @ coordinates
        
        # Ecuación (25): εf = ||(f-f̄) - fₚ||₂
        distance = np.linalg.norm(diff - face_projection)
        
        return distance
    
    def step_6c_classify_face(self, image_path: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Paso 6c: Clasificar la imagen según los umbrales.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Tupla (persona_reconocida, distancia_minima, distancias_por_persona)
        """
        if not self.is_trained:
            raise ValueError("El modelo debe estar entrenado primero")
        
        # Calcular distancia al espacio facial
        face_distance = self.step_6b_calculate_face_distance(image_path)
        
        # Si εf > ε₁, la imagen no es una cara
        if face_distance > self.epsilon_1:
            return None, face_distance, {}
          
        # Acá entraríamos al paso 7a (si εf <= ε₁)
        # Lo que me hace pensar que debería llamarse mejor 6c2_a o algo así
        # Sí, me dió pereza separar los pasos, pero bueno.
        
        # Calcular coordenadas
        coordinates = self.step_6a_calculate_coordinates(image_path)
        
        # Calcular distancias a cada persona conocida
        distances = {}
        min_distance = float('inf')
        recognized_person = None
        
        for person, person_coordinates_list in self.training_coordinates.items():
            person_distances = []
            for person_coords in person_coordinates_list:
                # Ecuación (22): εᵢ = ||x - xᵢ||₂
                distance = np.linalg.norm(coordinates - person_coords)
                person_distances.append(distance)
            
            # Usar la distancia mínima a esta persona
            min_person_distance = min(person_distances)
            distances[person] = min_person_distance
            
            if min_person_distance < min_distance:
                min_distance = min_person_distance
                recognized_person = person
        
        # Si εᵢ > ε₀ para todas las personas, es cara desconocida
        if min_distance > self.epsilon_0:
            recognized_person = "Cara desconocida"
        
        return recognized_person, min_distance, distances

    def _preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Preprocesa una imagen para el reconocimiento.
        
        Args:
            image_path: Ruta a la imagen
            target_size: Tamaño objetivo
            
        Returns:
            Imagen como vector 1D normalizado
        """
        # Cargar imagen
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Redimensionar
        image = cv2.resize(image, target_size)
        
        # Normalizar a [0, 1]
        image = image.astype(np.float64) / 255.0
        
        # Convertir a vector 1D
        return image.flatten()
        
    def train(self, training_data: Dict[str, List[str]], 
              epsilon_1: float = None, epsilon_0: float = None):
        """
        Ejecuta todos los pasos de entrenamiento del algoritmo SVD.
        
        Args:
            training_data: Diccionario {persona: [rutas_imagenes]}
            epsilon_1
            epsilon_0
        """
        print("=== ENTRENAMIENTO DEL MODELO SVD ===")
        
        # Paso 1: Obtener conjunto de entrenamiento
        S = self.step_1_obtain_training_set(training_data)
        
        # Paso 2: Calcular imagen media
        self.step_2_calculate_mean_face(S)
        
        # Paso 3: Formar matriz A
        A = self.step_3_form_matrix_A(S)
        
        # Paso 4: Calcular SVD
        self.step_4_calculate_svd(A)
        
        # Paso 5: Calcular coordenadas de entrenamiento
        self.step_5_calculate_training_coordinates(A)
        
        # Paso 6: Configurar umbrales
        self.step_5_a_b_choose_thresholds(epsilon_1, epsilon_0)
        
        self.is_trained = True
        print("=== ENTRENAMIENTO COMPLETADO ===")
    
    def recognize_face(self, image_path: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Método principal para reconocer una cara siguiendo todos los pasos.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Tupla (persona_reconocida, distancia_minima, distancias_por_persona)
        """
        return self.step_6c_classify_face(image_path)