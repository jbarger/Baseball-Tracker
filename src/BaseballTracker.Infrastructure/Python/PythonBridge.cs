using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using BaseballTracker.Core.Interfaces;
using BaseballTracker.Core.Models;
using Microsoft.Extensions.Logging;

namespace BaseballTracker.Infrastructure.Python;

/// <summary>
/// Bridge for calling Python CV services via HTTP
/// </summary>
public class PythonBridge : IPythonBridge
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<PythonBridge> _logger;
    private readonly JsonSerializerOptions _jsonOptions;

    public PythonBridge(HttpClient httpClient, ILogger<PythonBridge> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
        
        // Configure JSON options for snake_case conversion
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };
    }

    public async Task<bool> HealthCheckAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/health");
            
            if (response.IsSuccessStatusCode)
            {
                _logger.LogInformation("Python service health check: OK");
                return true;
            }
            
            _logger.LogWarning("Python service health check failed: {StatusCode}", 
                response.StatusCode);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Python service health check failed with exception");
            return false;
        }
    }

    public async Task<BallTrackingResult> TrackBallAsync(string videoPath, string cameraId = "default")
    {
        try
        {
            _logger.LogInformation("Requesting ball tracking for: {VideoPath}", videoPath);
            
            var request = new { video_path = videoPath, camera_id = cameraId };
            var response = await _httpClient.PostAsJsonAsync("/track/ball", request, _jsonOptions);
            
            response.EnsureSuccessStatusCode();
            
            var result = await response.Content.ReadFromJsonAsync<BallTrackingResult>(_jsonOptions);
            
            if (result == null)
            {
                throw new InvalidOperationException("Received null result from Python service");
            }
            
            _logger.LogInformation("Ball tracking complete: {ExitVelocity} mph, {LaunchAngle}°", 
                result.ExitVelocityMph, result.LaunchAngleDegrees);
            
            return result;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error during ball tracking for: {VideoPath}", videoPath);
            throw new Exception($"Failed to communicate with Python service: {ex.Message}", ex);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Ball tracking failed for: {VideoPath}", videoPath);
            throw;
        }
    }

    public async Task<BatTrackingResult> TrackBatAsync(string videoPath, string cameraId = "default")
    {
        try
        {
            _logger.LogInformation("Requesting bat tracking for: {VideoPath}", videoPath);
            
            var request = new { video_path = videoPath, camera_id = cameraId };
            var response = await _httpClient.PostAsJsonAsync("/track/bat", request, _jsonOptions);
            
            response.EnsureSuccessStatusCode();
            
            var result = await response.Content.ReadFromJsonAsync<BatTrackingResult>(_jsonOptions);
            
            if (result == null)
            {
                throw new InvalidOperationException("Received null result from Python service");
            }
            
            _logger.LogInformation("Bat tracking complete: {BatSpeed} mph", 
                result.BatSpeedMph);
            
            return result;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error during bat tracking for: {VideoPath}", videoPath);
            throw new Exception($"Failed to communicate with Python service: {ex.Message}", ex);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Bat tracking failed for: {VideoPath}", videoPath);
            throw;
        }
    }
}
