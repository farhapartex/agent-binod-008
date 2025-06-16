from typing import Dict, Optional
import requests

class WeatherTool:
    def __init__(self, api_key):
        self.name = "Weather"
        self.api_key=api_key
        self.description = "Get current weather information for a city"
        self.geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
        self.weather_url = "https://api.openweathermap.org/data/3.0/onecall"

    def get_coordinates(self, city: str) -> Optional[Dict[str, float]]:
        """Get latitude and longitude for a city using OpenWeatherMap Geocoding API"""
        try:
            if not self.api_key:
                # Use fallback coordinates if no API key
                raise Exception("Need Weather API Key")

            # First try geocoding API
            params = {
                "q": city,
                "limit": 1,
                "appid": self.api_key
            }

            response = requests.get(self.geocoding_url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data:
                    return {
                        "lat": data[0]["lat"],
                        "lon": data[0]["lon"]
                    }

            raise Exception(f"API Error with status: {response.status_code}")

        except Exception as e:
            print(f"{e}")

    def get_weather(self, city: str) -> str:
        """Get weather information for a city using OpenWeatherMap API"""
        try:
            # Get coordinates
            coords = self.get_coordinates(city)
            if not coords:
                raise Exception(f"Coordinates not found")

            if not self.api_key:
                raise Exception("Need Weather API Key")

            # Get weather data
            params = {
                "lat": coords["lat"],
                "lon": coords["lon"],
                "exclude": "minutely,alerts",
                "appid": self.api_key,
                "units": "metric"
            }

            response = requests.get(self.weather_url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return self._format_weather_response(city, data)
            elif response.status_code == 401:
                return "Invalid OpenWeatherMap API key. Please check your API key."
            elif response.status_code == 429:
                return "API rate limit exceeded. Please try again later."
            else:
                return f"Weather API error: {response.status_code}"

        except requests.exceptions.Timeout:
            return f"❌ Weather service timeout for {city}. Please try again."
        except requests.exceptions.ConnectionError:
            return f"❌ Could not connect to weather service for {city}."
        except Exception as e:
            return f"❌ Weather error for {city}: {str(e)}"

    def _format_weather_response(self, city: str, data: Dict) -> str:
        """Format the weather API response into a readable string"""
        try:
            current = data.get("current", {})
            hourly = data.get("hourly", [])[:3]  # Next 3 hours
            daily = data.get("daily", [])[:3]  # Next 3 days

            # Current weather
            temp = current.get("temp", 0)
            feels_like = current.get("feels_like", 0)
            humidity = current.get("humidity", 0)
            description = current.get("weather", [{}])[0].get("description", "Unknown")
            wind_speed = current.get("wind_speed", 0)

            response = f"🌤️ **Weather for {city.title()}**\n"
            response += f"**Current:** {temp:.1f}°C (feels like {feels_like:.1f}°C)\n"
            response += f"**Condition:** {description.title()}\n"
            response += f"**Humidity:** {humidity}% | **Wind:** {wind_speed:.1f} m/s\n"

            # Hourly forecast
            if hourly:
                response += f"\n**Next few hours:**\n"
                for i, hour in enumerate(hourly):
                    hour_temp = hour.get("temp", 0)
                    hour_desc = hour.get("weather", [{}])[0].get("description", "")
                    response += f"  +{i + 1}h: {hour_temp:.1f}°C, {hour_desc}\n"

            # Daily forecast
            if daily:
                response += f"\n**Next few days:**\n"
                for i, day in enumerate(daily):
                    day_temp_max = day.get("temp", {}).get("max", 0)
                    day_temp_min = day.get("temp", {}).get("min", 0)
                    day_desc = day.get("weather", [{}])[0].get("description", "")
                    response += f"  Day {i + 1}: {day_temp_min:.1f}°C - {day_temp_max:.1f}°C, {day_desc}\n"

            return response.strip()

        except Exception as e:
            return f"Error formatting weather data: {str(e)}"

    def run(self, city: str) -> str:
        """Simulate weather API call"""
        return self.get_weather(city)