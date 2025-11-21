"""
Modelo de optimización para asignación óptima de brigadas
Basado en Facility Location / p-median problem con pesos probabilísticos
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from ortools.linear_solver import pywraplp
import pulp

from config import OPTIMIZATION_CONFIG, GEO_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResourceAllocationOptimizer:
    """
    Optimizador para asignación óptima de brigadas contra incendios
    """
    
    def __init__(self, max_brigades: int = None, max_bases: int = None,
                 solver: str = "PULP_CBC_CMD"):
        """
        Args:
            max_brigades: Número máximo de brigadas disponibles
            max_bases: Número máximo de bases posibles
            solver: Solver a usar ('CBC', 'GLPK', 'CPLEX')
        """
        config = OPTIMIZATION_CONFIG
        self.max_brigades = max_brigades or config['max_brigades']
        self.max_bases = max_bases or config['max_bases']
        self.solver_name = solver
        self.config = config
        
        self.risk_map = None
        self.base_locations = None
        self.travel_times = None
        self.solution = None
    
    def prepare_data(self, risk_map: pd.DataFrame,
                    comuna_coords: Optional[pd.DataFrame] = None,
                    base_locations: Optional[pd.DataFrame] = None):
        """
        Prepara datos para optimización
        
        Args:
            risk_map: DataFrame con riesgo por comuna (debe tener 'comuna', 'riesgo_probabilidad', 
                     y opcionalmente 'severidad_esperada')
            comuna_coords: DataFrame con coordenadas de comunas (columnas: 'comuna', 'lat', 'lon')
            base_locations: DataFrame con ubicaciones de bases posibles (columnas: 'base', 'lat', 'lon')
        """
        logger.info("Preparando datos para optimización...")
        
        self.risk_map = risk_map.copy()
        
        # Generar ubicaciones si no se proporcionan
        if comuna_coords is None:
            logger.warning("No se proporcionaron coordenadas de comunas. Generando coordenadas sintéticas.")
            self.comuna_coords = self._generate_synthetic_coords(
                comunas=risk_map['comuna'].unique(),
                coord_type='comuna'
            )
        else:
            self.comuna_coords = comuna_coords.copy()
        
        if base_locations is None:
            logger.warning("No se proporcionaron ubicaciones de bases. Generando bases sintéticas.")
            self.base_locations = self._generate_synthetic_coords(
                comunas=risk_map['comuna'].unique(),
                coord_type='base',
                n_bases=self.max_bases
            )
        else:
            self.base_locations = base_locations.copy()
        
        # Calcular tiempos de viaje
        self.travel_times = self._calculate_travel_times(
            self.comuna_coords, self.base_locations
        )
        
        logger.info(f"Datos preparados: {len(self.risk_map)} comunas, {len(self.base_locations)} bases")
    
    def _generate_synthetic_coords(self, comunas: np.ndarray,
                                  coord_type: str = 'comuna',
                                  n_bases: int = None) -> pd.DataFrame:
        """Genera coordenadas sintéticas para comunas o bases"""
        if coord_type == 'comuna':
            # Coordenadas aproximadas de Chile (latitud entre -17 y -56, longitud entre -110 y -66)
            n = len(comunas)
            lats = np.random.uniform(-35, -40, n)  # Región central de Chile
            lons = np.random.uniform(-73, -71, n)
            
            df = pd.DataFrame({
                'comuna': comunas,
                'lat': lats,
                'lon': lons
            })
        else:  # base
            # Generar bases distribuidas estratégicamente
            if n_bases is None:
                n_bases = min(len(comunas), self.max_bases)
            
            # Seleccionar comunas estratégicas (comunas con más riesgo o distribuidas)
            selected_comunas = np.random.choice(
                comunas, 
                size=min(n_bases, len(comunas)), 
                replace=False
            )
            
            lats = np.random.uniform(-35, -40, n_bases)
            lons = np.random.uniform(-73, -71, n_bases)
            
            df = pd.DataFrame({
                'base': [f'Base_{i+1}' for i in range(n_bases)],
                'lat': lats,
                'lon': lons
            })
        
        return df
    
    def _check_solver_availability(self, solver_name: str) -> bool:
        """
        Verifica si un solver está disponible
        
        Args:
            solver_name: Nombre del solver a verificar
            
        Returns:
            True si el solver está disponible, False en caso contrario
        """
        try:
            solver = pulp.getSolver(solver_name, msg=0)
            # Intentar crear un problema simple para verificar
            test_prob = pulp.LpProblem("test", pulp.LpMinimize)
            x = pulp.LpVariable("x", lowBound=0)
            test_prob += x
            test_prob.solve(solver)
            return True
        except Exception as e:
            logger.debug(f"Solver {solver_name} no disponible: {e}")
            return False
    
    def _solve_with_fallback(self, prob: pulp.LpProblem, preferred_solver: str) -> Dict:
        """
        Intenta resolver el problema con el solver preferido, 
        y usa alternativas si falla
        
        Args:
            prob: Problema de optimización PuLP
            preferred_solver: Solver preferido
            
        Returns:
            Dict con 'status' y 'solver_used'
        """
        # Lista de solvers a probar en orden de preferencia
        solver_options = [
            preferred_solver,
            "HiGHS_CMD",  # HiGHS suele ser más confiable que CBC
            "PULP_CBC_CMD",
            "COIN_CMD",
            "PULP_CBC",
            None  # Default solver (PuLP intenta automáticamente) - se maneja en el loop
        ]
        
        # Si preferred_solver es None, usar el default de PuLP
        if preferred_solver is None:
            solver_options = ["HiGHS_CMD", "PULP_CBC_CMD", "COIN_CMD", "PULP_CBC", None]
        
        # Eliminar duplicados manteniendo el orden
        seen = set()
        unique_solvers = []
        for s in solver_options:
            if s not in seen:
                seen.add(s)
                unique_solvers.append(s)
        
        last_error = None
        
        for solver_name in unique_solvers:
            try:
                if solver_name is None:
                    logger.info("Intentando resolver con solver por defecto de PuLP...")
                    status = prob.solve()
                    solver_used = "default"
                else:
                    logger.info(f"Intentando resolver con solver: {solver_name}...")
                    solver = pulp.getSolver(solver_name, msg=1)
                    status = prob.solve(solver)
                    solver_used = solver_name
                
                logger.info(f"Solver {solver_used} retornó status: {pulp.LpStatus[status]}")
                
                return {
                    'status': status,
                    'solver_used': solver_used
                }
                
            except pulp.apis.core.PulpSolverError as e:
                last_error = e
                logger.warning(f"Error con solver {solver_name}: {e}")
                continue
            except Exception as e:
                last_error = e
                logger.warning(f"Error inesperado con solver {solver_name}: {e}")
                continue
        
        # Si todo falla, lanzar error informativo
        solver_names_tried = [s for s in unique_solvers if s is not None]
        error_msg = (
            f"No se pudo resolver el problema de optimización. "
            f"Se intentaron los siguientes solvers: {', '.join(solver_names_tried) if solver_names_tried else 'ninguno'}. "
            f"Último error: {last_error}. "
            f"Verifica que tengas instalado al menos uno de: CBC, COIN-OR, o HiGHS. "
            f"Instalación sugerida: pip install pulp[coin] o pip install pulp[highs]"
        )
        raise RuntimeError(error_msg)
    
    def _calculate_travel_times(self, comuna_coords: pd.DataFrame,
                               base_coords: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula tiempos de viaje entre comunas y bases
        Usa distancia euclidiana como proxy (en la realidad usarías rutas reales)
        """
        travel_times_list = []
        
        for _, comuna_row in comuna_coords.iterrows():
            comuna = comuna_row['comuna']
            comuna_lat = comuna_row['lat']
            comuna_lon = comuna_row['lon']
            
            for _, base_row in base_coords.iterrows():
                base = base_row['base']
                base_lat = base_row['lat']
                base_lon = base_row['lon']
                
                # Distancia euclidiana (simplificado)
                # En la realidad usarías geopy o calcular rutas reales
                distance_km = np.sqrt(
                    (comuna_lat - base_lat)**2 * 111**2 +  # 1 grado lat ≈ 111 km
                    (comuna_lon - base_lon)**2 * 111**2 * np.cos(np.radians(comuna_lat))
                )
                
                # Tiempo de viaje (suponiendo velocidad promedio de 60 km/h)
                travel_time_minutes = (distance_km / 60) * 60
                
                # Agregar variabilidad
                travel_time_minutes *= np.random.uniform(0.8, 1.2)
                
                travel_times_list.append({
                    'comuna': comuna,
                    'base': base,
                    'tiempo_minutos': travel_time_minutes,
                    'distancia_km': distance_km
                })
        
        return pd.DataFrame(travel_times_list)
    
    def optimize(self, objective: str = None,
                risk_weight: float = None,
                severity_weight: float = None) -> Dict:
        """
        Resuelve el problema de optimización
        
        Args:
            objective: Tipo de objetivo ('minimize_damage', 'minimize_response_time')
            risk_weight: Peso del riesgo en la función objetivo
            severity_weight: Peso de la severidad esperada
        
        Returns:
            Dict con solución y estadísticas
        """
        logger.info("Iniciando optimización...")
        
        if self.risk_map is None or self.travel_times is None:
            raise ValueError("Debes llamar a prepare_data() primero")
        
        objective = objective or self.config['objective']
        risk_weight = risk_weight or self.config['risk_weight']
        severity_weight = severity_weight or self.config['severity_weight']
        
        # Preparar datos para el modelo
        comunas = self.risk_map['comuna'].unique()
        bases = self.base_locations['base'].unique()
        
        # Crear diccionarios con riesgos y severidades
        risk_dict = dict(zip(
            self.risk_map['comuna'],
            self.risk_map['riesgo_probabilidad']
        ))
        
        # Severidad esperada (área quemada o personas afectadas)
        if 'severidad_esperada' in self.risk_map.columns:
            severity_dict = dict(zip(
                self.risk_map['comuna'],
                self.risk_map['severidad_esperada']
            ))
        else:
            # Si no hay severidad, usar riesgo como proxy
            severity_dict = risk_dict.copy()
        
        # Diccionario de tiempos de viaje
        time_dict = {}
        for _, row in self.travel_times.iterrows():
            time_dict[(row['comuna'], row['base'])] = row['tiempo_minutos']
        
        # Intentar resolver con diferentes métodos en orden de preferencia
        solution = None
        method_used = None
        
        # 1. Intentar con PuLP
        try:
            logger.info("Intentando optimización con PuLP...")
            solution = self._solve_with_pulp(
                comunas, bases, risk_dict, severity_dict, time_dict,
                objective, risk_weight, severity_weight
            )
            method_used = "PuLP"
        except Exception as e:
            logger.warning(f"PuLP falló: {e}")
            # 2. Intentar con OR-Tools
            try:
                logger.info("Intentando optimización con OR-Tools...")
                solution = self._solve_with_ortools(
                    comunas, bases, risk_dict, severity_dict, time_dict,
                    objective, risk_weight, severity_weight
                )
                method_used = "OR-Tools"
            except Exception as e2:
                logger.warning(f"OR-Tools falló: {e2}")
                # 3. Usar método heurístico como último recurso
                logger.info("Usando método heurístico como último recurso...")
                solution = self._solve_heuristic(
                    comunas, bases, risk_dict, severity_dict, time_dict,
                    objective, risk_weight, severity_weight
                )
                method_used = "Heurístico"
        
        if solution is None:
            raise RuntimeError("No se pudo resolver el problema de optimización con ningún método disponible.")
        
        solution['method_used'] = method_used
        self.solution = solution
        
        logger.info(f"Optimización completada usando {method_used}. Brigadas asignadas: {solution['total_brigades']}")
        
        return solution
    
    def _solve_with_pulp(self, comunas: List, bases: List,
                        risk_dict: Dict, severity_dict: Dict, time_dict: Dict,
                        objective: str, risk_weight: float, severity_weight: float) -> Dict:
        """
        Resuelve el problema usando PuLP
        
        Variables de decisión:
        - y_b: 1 si se activa la base b
        - x_b,i: 1 si la comuna i es cubierta por la base b
        """
        # Crear problema
        prob = pulp.LpProblem("Asignacion_Brigadas", pulp.LpMinimize)
        
        # Variables de decisión
        # y_b: si la base b está activa
        y = pulp.LpVariable.dicts("base_activa", bases, cat='Binary')
        
        # x_b,i: si la comuna i es cubierta por la base b
        x = pulp.LpVariable.dicts("cobertura",
                                  [(b, i) for b in bases for i in comunas],
                                  cat='Binary')
        
        # n_b: número de brigadas en la base b (entero, entre 0 y max_brigades)
        n = pulp.LpVariable.dicts("brigadas_por_base", bases,
                                  lowBound=0, upBound=self.max_brigades,
                                  cat='Integer')
        
        # Función objetivo
        if objective == "minimize_damage":
            # Minimizar costo esperado = riesgo * severidad * tiempo_respuesta
            prob += pulp.lpSum([
                risk_weight * risk_dict[i] * severity_weight * severity_dict[i] * 
                time_dict.get((i, b), 999) * x[(b, i)]
                for b in bases for i in comunas
            ])
        else:  # minimize_response_time
            # Minimizar tiempo de respuesta esperado
            prob += pulp.lpSum([
                risk_weight * risk_dict[i] * time_dict.get((i, b), 999) * x[(b, i)]
                for b in bases for i in comunas
            ])
        
        # Restricciones
        
        # 1. Cada comuna debe estar cubierta por al menos una base
        for i in comunas:
            prob += pulp.lpSum([x[(b, i)] for b in bases]) >= 1
        
        # 2. Solo se puede cubrir desde bases activas
        for b in bases:
            for i in comunas:
                prob += x[(b, i)] <= y[b]
        
        # 3. Número máximo de bases activas
        prob += pulp.lpSum([y[b] for b in bases]) <= self.max_bases
        
        # 4. Número total de brigadas limitado
        prob += pulp.lpSum([n[b] for b in bases]) <= self.max_brigades
        
        # 5. Solo se pueden asignar brigadas a bases activas
        for b in bases:
            prob += n[b] <= y[b] * self.max_brigades
        
        # 6. Si una base está activa, debe tener al menos 1 brigada
        for b in bases:
            prob += n[b] >= y[b]
        
        # Resolver con manejo de errores y fallback
        solver_result = self._solve_with_fallback(prob, self.solver_name)
        if solver_result['status'] != pulp.LpStatusOptimal:
            logger.warning(f"El solver retornó status: {solver_result['status']}")
            if solver_result['status'] == pulp.LpStatusInfeasible:
                raise ValueError("El problema es infactible. Verifica las restricciones.")
            elif solver_result['status'] == pulp.LpStatusUnbounded:
                raise ValueError("El problema es ilimitado. Verifica la función objetivo.")
        
        # Extraer solución
        bases_activas = [b for b in bases if y[b].varValue == 1]
        brigadas_por_base = {b: int(n[b].varValue) for b in bases if n[b].varValue is not None and n[b].varValue > 0}
        cobertura = {(b, i): x[(b, i)].varValue for b in bases for i in comunas 
                    if x[(b, i)].varValue is not None and x[(b, i)].varValue == 1}
        
        # Calcular estadísticas
        total_brigades = sum(brigadas_por_base.values())
        
        # Tiempo promedio de respuesta esperado
        tiempos_respuesta = []
        for i in comunas:
            bases_que_cubren = [b for b in bases if cobertura.get((b, i)) == 1]
            if bases_que_cubren:
                tiempo_min = min([time_dict.get((i, b), 999) for b in bases_que_cubren])
                tiempos_respuesta.append({
                    'comuna': i,
                    'tiempo_respuesta_min': tiempo_min,
                    'riesgo': risk_dict[i]
                })
        
        tiempo_promedio = np.mean([t['tiempo_respuesta_min'] for t in tiempos_respuesta])
        tiempo_promedio_ponderado = np.average(
            [t['tiempo_respuesta_min'] for t in tiempos_respuesta],
            weights=[t['riesgo'] for t in tiempos_respuesta]
        )
        
        # Obtener el status del problema después de resolver
        problem_status = prob.status
        
        solution = {
            'bases_activas': bases_activas,
            'brigadas_por_base': brigadas_por_base,
            'cobertura': cobertura,
            'total_brigades': total_brigades,
            'total_bases_activas': len(bases_activas),
            'tiempo_respuesta_promedio': tiempo_promedio,
            'tiempo_respuesta_ponderado': tiempo_promedio_ponderado,
            'status': pulp.LpStatus[problem_status],
            'objective_value': pulp.value(prob.objective) if problem_status == pulp.LpStatusOptimal else None
        }
        
        return solution
    
    def _solve_with_ortools(self, comunas: List, bases: List,
                            risk_dict: Dict, severity_dict: Dict, time_dict: Dict,
                            objective: str, risk_weight: float, severity_weight: float) -> Dict:
        """
        Resuelve el problema usando OR-Tools como alternativa a PuLP
        """
        from ortools.linear_solver import pywraplp
        
        # Crear solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            # Si SCIP no está disponible, intentar con GLOP (solo continuo) o CBC
            solver = pywraplp.Solver.CreateSolver('CBC')
            if not solver:
                raise RuntimeError("No se pudo crear solver de OR-Tools")
        
        # Variables de decisión
        # y_b: si la base b está activa
        y = {}
        for b in bases:
            y[b] = solver.IntVar(0, 1, f'base_activa_{b}')
        
        # x_b,i: si la comuna i es cubierta por la base b
        x = {}
        for b in bases:
            for i in comunas:
                x[(b, i)] = solver.IntVar(0, 1, f'cobertura_{b}_{i}')
        
        # n_b: número de brigadas en la base b
        n = {}
        for b in bases:
            n[b] = solver.IntVar(0, self.max_brigades, f'brigadas_{b}')
        
        # Función objetivo
        if objective == "minimize_damage":
            objective_expr = solver.Objective()
            for b in bases:
                for i in comunas:
                    cost = (risk_weight * risk_dict[i] * severity_weight * 
                           severity_dict[i] * time_dict.get((i, b), 999))
                    objective_expr.SetCoefficient(x[(b, i)], cost)
            objective_expr.SetMinimization()
        else:  # minimize_response_time
            objective_expr = solver.Objective()
            for b in bases:
                for i in comunas:
                    cost = risk_weight * risk_dict[i] * time_dict.get((i, b), 999)
                    objective_expr.SetCoefficient(x[(b, i)], cost)
            objective_expr.SetMinimization()
        
        # Restricciones
        # 1. Cada comuna debe estar cubierta por al menos una base
        for i in comunas:
            solver.Add(sum([x[(b, i)] for b in bases]) >= 1)
        
        # 2. Solo se puede cubrir desde bases activas
        for b in bases:
            for i in comunas:
                solver.Add(x[(b, i)] <= y[b])
        
        # 3. Número máximo de bases activas
        solver.Add(sum([y[b] for b in bases]) <= self.max_bases)
        
        # 4. Número total de brigadas limitado
        solver.Add(sum([n[b] for b in bases]) <= self.max_brigades)
        
        # 5. Solo se pueden asignar brigadas a bases activas
        for b in bases:
            solver.Add(n[b] <= y[b] * self.max_brigades)
        
        # 6. Si una base está activa, debe tener al menos 1 brigada
        for b in bases:
            solver.Add(n[b] >= y[b])
        
        # Resolver
        status = solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError(f"OR-Tools no pudo resolver el problema. Status: {status}")
        
        # Extraer solución
        bases_activas = [b for b in bases if y[b].solution_value() > 0.5]
        brigadas_por_base = {b: int(n[b].solution_value()) for b in bases 
                            if n[b].solution_value() > 0}
        cobertura = {(b, i): x[(b, i)].solution_value() for b in bases for i in comunas 
                    if x[(b, i)].solution_value() > 0.5}
        
        # Calcular estadísticas
        total_brigades = sum(brigadas_por_base.values())
        
        tiempos_respuesta = []
        for i in comunas:
            bases_que_cubren = [b for b in bases if cobertura.get((b, i), 0) > 0.5]
            if bases_que_cubren:
                tiempo_min = min([time_dict.get((i, b), 999) for b in bases_que_cubren])
                tiempos_respuesta.append({
                    'comuna': i,
                    'tiempo_respuesta_min': tiempo_min,
                    'riesgo': risk_dict[i]
                })
        
        tiempo_promedio = np.mean([t['tiempo_respuesta_min'] for t in tiempos_respuesta]) if tiempos_respuesta else 0
        tiempo_promedio_ponderado = np.average(
            [t['tiempo_respuesta_min'] for t in tiempos_respuesta],
            weights=[t['riesgo'] for t in tiempos_respuesta]
        ) if tiempos_respuesta else 0
        
        solution = {
            'bases_activas': bases_activas,
            'brigadas_por_base': brigadas_por_base,
            'cobertura': cobertura,
            'total_brigades': total_brigades,
            'total_bases_activas': len(bases_activas),
            'tiempo_respuesta_promedio': tiempo_promedio,
            'tiempo_respuesta_ponderado': tiempo_promedio_ponderado,
            'status': 'Optimal',
            'objective_value': solver.Objective().Value()
        }
        
        return solution
    
    def _solve_heuristic(self, comunas: List, bases: List,
                        risk_dict: Dict, severity_dict: Dict, time_dict: Dict,
                        objective: str, risk_weight: float, severity_weight: float) -> Dict:
        """
        Método heurístico simple que siempre funciona como último recurso
        Basado en algoritmo greedy: selecciona bases que minimizan el costo esperado
        """
        logger.warning("Usando método heurístico. La solución puede no ser óptima.")
        
        # Calcular score para cada base
        base_scores = {}
        for b in bases:
            score = 0
            comunas_cubiertas = 0
            
            for i in comunas:
                tiempo = time_dict.get((i, b), 999)
                if tiempo < 999:  # Base puede cubrir esta comuna
                    comunas_cubiertas += 1
                    # Calcular costo esperado
                    if objective == "minimize_damage":
                        costo = (risk_weight * risk_dict[i] * severity_weight * 
                                severity_dict[i] * tiempo)
                    else:
                        costo = risk_weight * risk_dict[i] * tiempo
                    score += costo
            
            if comunas_cubiertas > 0:
                # Score promedio normalizado por comunas cubiertas
                base_scores[b] = score / comunas_cubiertas
            else:
                base_scores[b] = float('inf')
        
        # Ordenar bases por score (menor es mejor)
        bases_ordenadas = sorted(base_scores.items(), key=lambda x: x[1])
        
        # Seleccionar hasta max_bases bases
        bases_seleccionadas = []
        comunas_cubiertas = set()
        
        for b, _ in bases_ordenadas[:self.max_bases]:
            if base_scores[b] == float('inf'):
                continue  # Saltar bases que no pueden cubrir ninguna comuna
            
            bases_seleccionadas.append(b)
            
            # Marcar comunas que puede cubrir esta base
            for i in comunas:
                tiempo = time_dict.get((i, b), 999)
                if tiempo < 999:
                    comunas_cubiertas.add(i)
            
            # Si todas las comunas están cubiertas, parar
            if len(comunas_cubiertas) == len(comunas):
                break
        
        # Asegurar que haya al menos una base seleccionada
        if not bases_seleccionadas:
            # Si no hay bases válidas, usar la primera disponible
            if bases_ordenadas:
                bases_seleccionadas = [bases_ordenadas[0][0]]
            else:
                # Si no hay bases en absoluto, usar la primera base
                if bases:
                    bases_seleccionadas = [bases[0]]
        
        # Asignar cobertura: cada comuna a la base más cercana disponible
        cobertura = {}
        bases_activas = bases_seleccionadas
        
        for i in comunas:
            mejor_base = None
            mejor_tiempo = float('inf')
            
            for b in bases_activas:
                tiempo = time_dict.get((i, b), 999)
                if tiempo < mejor_tiempo:
                    mejor_tiempo = tiempo
                    mejor_base = b
            
            # Asegurar que cada comuna esté asignada
            if mejor_base:
                cobertura[(mejor_base, i)] = 1
            elif bases_activas:
                # Si no se encontró base cercana, asignar a la primera base disponible
                cobertura[(bases_activas[0], i)] = 1
        
        # Distribuir brigadas: proporcional al riesgo cubierto por cada base
        total_riesgo = sum(risk_dict.values())
        brigadas_por_base = {}
        
        for b in bases_activas:
            riesgo_cubierto = sum([risk_dict[i] for i in comunas 
                                  if cobertura.get((b, i)) == 1])
            proporcion = riesgo_cubierto / total_riesgo if total_riesgo > 0 else 0
            brigadas = max(1, int(proporcion * self.max_brigades))
            brigadas_por_base[b] = brigadas
        
        # Ajustar para no exceder max_brigades
        total_brigadas = sum(brigadas_por_base.values())
        if total_brigadas > self.max_brigades:
            factor = self.max_brigades / total_brigadas
            for b in brigadas_por_base:
                brigadas_por_base[b] = max(1, int(brigadas_por_base[b] * factor))
        
        total_brigadas = sum(brigadas_por_base.values())
        
        # Calcular estadísticas
        tiempos_respuesta = []
        for i in comunas:
            bases_que_cubren = [b for b in bases_activas if cobertura.get((b, i)) == 1]
            if bases_que_cubren:
                tiempo_min = min([time_dict.get((i, b), 999) for b in bases_que_cubren])
                tiempos_respuesta.append({
                    'comuna': i,
                    'tiempo_respuesta_min': tiempo_min,
                    'riesgo': risk_dict[i]
                })
        
        tiempo_promedio = np.mean([t['tiempo_respuesta_min'] for t in tiempos_respuesta]) if tiempos_respuesta else 0
        tiempo_promedio_ponderado = np.average(
            [t['tiempo_respuesta_min'] for t in tiempos_respuesta],
            weights=[t['riesgo'] for t in tiempos_respuesta]
        ) if tiempos_respuesta else 0
        
        # Calcular valor objetivo aproximado
        objective_value = 0
        for (b, i), val in cobertura.items():
            if val == 1:
                tiempo = time_dict.get((i, b), 999)
                if objective == "minimize_damage":
                    objective_value += (risk_weight * risk_dict[i] * severity_weight * 
                                      severity_dict[i] * tiempo)
                else:
                    objective_value += risk_weight * risk_dict[i] * tiempo
        
        solution = {
            'bases_activas': bases_activas,
            'brigadas_por_base': brigadas_por_base,
            'cobertura': cobertura,
            'total_brigades': total_brigadas,
            'total_bases_activas': len(bases_activas),
            'tiempo_respuesta_promedio': tiempo_promedio,
            'tiempo_respuesta_ponderado': tiempo_promedio_ponderado,
            'status': 'Heuristic',
            'objective_value': objective_value
        }
        
        return solution
    
    def get_allocation_map(self) -> pd.DataFrame:
        """
        Genera mapa de asignación de brigadas por comuna
        
        Returns:
            DataFrame con asignación de brigadas por comuna
        """
        if self.solution is None:
            raise ValueError("Debes ejecutar optimize() primero")
        
        allocation_list = []
        
        for comuna in self.risk_map['comuna'].unique():
            # Encontrar base que cubre esta comuna
            base_asignada = None
            for (b, i), covered in self.solution['cobertura'].items():
                if i == comuna and covered == 1:
                    base_asignada = b
                    break
            
            if base_asignada:
                tiempo_respuesta = self.travel_times[
                    (self.travel_times['comuna'] == comuna) &
                    (self.travel_times['base'] == base_asignada)
                ]['tiempo_minutos'].values[0]
                
                allocation_list.append({
                    'comuna': comuna,
                    'base_asignada': base_asignada,
                    'brigadas_en_base': self.solution['brigadas_por_base'].get(base_asignada, 0),
                    'tiempo_respuesta_min': tiempo_respuesta,
                    'riesgo': self.risk_map[self.risk_map['comuna'] == comuna]['riesgo_probabilidad'].values[0]
                })
        
        allocation_df = pd.DataFrame(allocation_list)
        
        # Agregar coordenadas de comunas y bases
        allocation_df = allocation_df.merge(
            self.comuna_coords[['comuna', 'lat', 'lon']],
            on='comuna',
            how='left'
        )
        allocation_df = allocation_df.merge(
            self.base_locations[['base', 'lat', 'lon']].rename(
                columns={'lat': 'base_lat', 'lon': 'base_lon'}
            ),
            left_on='base_asignada',
            right_on='base',
            how='left'
        )
        
        return allocation_df
    
    def compare_solutions(self, solution1: Dict, solution2: Dict) -> pd.DataFrame:
        """Compara dos soluciones de optimización"""
        comparison = {
            'Métrica': [
                'Bases activas',
                'Total brigadas',
                'Tiempo respuesta promedio (min)',
                'Tiempo respuesta ponderado (min)',
                'Valor objetivo'
            ],
            'Solución 1': [
                solution1['total_bases_activas'],
                solution1['total_brigades'],
                f"{solution1['tiempo_respuesta_promedio']:.2f}",
                f"{solution1['tiempo_respuesta_ponderado']:.2f}",
                f"{solution1['objective_value']:.2f}"
            ],
            'Solución 2': [
                solution2['total_bases_activas'],
                solution2['total_brigades'],
                f"{solution2['tiempo_respuesta_promedio']:.2f}",
                f"{solution2['tiempo_respuesta_ponderado']:.2f}",
                f"{solution2['objective_value']:.2f}"
            ]
        }
        
        return pd.DataFrame(comparison)

