# ─────────────────────────────────────────────
# tomtom_api.py — Wrapper per la TomTom Traffic Flow API
# ─────────────────────────────────────────────

import requests
import time
from config import API_KEY, TOMTOM_FLOW_ZOOM, TOMTOM_FLOW_UNIT

BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute"

def get_flow_segment(lat: float, lon: float, retries: int = 3) -> dict | None:
    """
    Chiama la TomTom Flow API per un punto geografico.

    Parametri:
        lat, lon : coordinate del punto (es. centro dell'edge)
        retries  : numero di tentativi in caso di errore

    Ritorna un dict con:
        current_speed      (km/h)
        free_flow_speed    (km/h)
        current_travel_time (secondi)
        free_flow_travel_time (secondi)
        confidence         (0.0 - 1.0)
        road_closure       (bool)
    Ritorna None se la chiamata fallisce.
    """
    url = f"{BASE_URL}/{TOMTOM_FLOW_ZOOM}/json"
    params = {
        "point": f"{lat},{lon}",
        "unit":  TOMTOM_FLOW_UNIT,
        "key":   API_KEY,
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json().get("flowSegmentData", {})
                return {
                    "current_speed":         data.get("currentSpeed", 0),
                    "free_flow_speed":        data.get("freeFlowSpeed", 0),
                    "current_travel_time":    data.get("currentTravelTime", 0),
                    "free_flow_travel_time":  data.get("freeFlowTravelTime", 0),
                    "confidence":             data.get("confidence", 0),
                    "road_closure":           data.get("roadClosure", False),
                }

            elif response.status_code == 429:
                # Rate limit raggiunto — aspetta e riprova
                print(f"  [!] Rate limit raggiunto, attendo 5s... (tentativo {attempt+1}/{retries})")
                time.sleep(5)

            elif response.status_code == 404:
                # Nessun dato per questo punto (strada non coperta)
                print(f"  [!] Nessun dato TomTom per lat={lat}, lon={lon}")
                return None

            else:
                print(f"  [!] Errore API {response.status_code}: {response.text[:100]}")
                return None

        except requests.exceptions.Timeout:
            print(f"  [!] Timeout, tentativo {attempt+1}/{retries}")
            time.sleep(2)
        except requests.exceptions.ConnectionError as e:
            print(f"  [!] Errore connessione: {e}")
            return None

    print(f"  [!] Tutti i tentativi falliti per lat={lat}, lon={lon}")
    return None


def get_flow_for_edge(from_lat: float, from_lon: float,
                      to_lat: float,   to_lon: float) -> dict | None:
    """
    Recupera i dati di flusso per un edge usando il suo punto medio.
    Se il punto medio non restituisce dati, prova start e fine dell'edge.
    """
    mid_lat = (from_lat + to_lat) / 2
    mid_lon = (from_lon + to_lon) / 2

    result = get_flow_segment(mid_lat, mid_lon)

    # Fallback: prova il punto iniziale
    if result is None:
        print(f"  [>] Fallback su punto iniziale dell'edge")
        result = get_flow_segment(from_lat, from_lon)

    # Fallback: prova il punto finale
    if result is None:
        print(f"  [>] Fallback su punto finale dell'edge")
        result = get_flow_segment(to_lat, to_lon)

    return result


def test_connection() -> bool:
    """
    Verifica che la API key funzioni con una chiamata di test.
    Usa le coordinate del centro dell'incrocio simulato.
    """
    print("Test connessione TomTom API...")
    # Centroide del tuo incrocio (Altamura, BA)
    result = get_flow_segment(41.099239, 16.881719)
    if result:
        print(f"  ✓ Connessione OK")
        print(f"  ✓ Velocità attuale: {result['current_speed']} km/h")
        print(f"  ✓ Velocità free-flow: {result['free_flow_speed']} km/h")
        print(f"  ✓ Confidence: {result['confidence']}")
        return True
    else:
        print("  ✗ Connessione fallita — verifica la API key in config.py")
        return False


if __name__ == "__main__":
    test_connection()