using Microsoft.AspNetCore.Mvc;
using BaseballTracker.Core.Interfaces;
using BaseballTracker.Core.Models;

namespace BaseballTracker.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class SwingsController : ControllerBase
{
    private readonly ISwingRepository _swingRepository;
    private readonly IPythonBridge _pythonBridge;
    private readonly ILogger<SwingsController> _logger;

    public SwingsController(
        ISwingRepository swingRepository,
        IPythonBridge pythonBridge,
        ILogger<SwingsController> logger)
    {
        _swingRepository = swingRepository;
        _pythonBridge = pythonBridge;
        _logger = logger;
    }

    /// <summary>
    /// Get a specific swing by ID
    /// </summary>
    [HttpGet("{id}")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<ActionResult<SwingData>> GetSwing(Guid id)
    {
        var swing = await _swingRepository.GetSwingAsync(id);
        
        if (swing == null)
            return NotFound();
        
        return Ok(swing);
    }

    /// <summary>
    /// Get all swings for a session
    /// </summary>
    [HttpGet("session/{sessionId}")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public async Task<ActionResult<List<SwingData>>> GetSessionSwings(Guid sessionId)
    {
        var swings = await _swingRepository.GetSwingsBySessionAsync(sessionId);
        return Ok(swings);
    }

    /// <summary>
    /// Get all swings for a player
    /// </summary>
    [HttpGet("player/{playerId}")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public async Task<ActionResult<List<SwingData>>> GetPlayerSwings(Guid playerId)
    {
        var swings = await _swingRepository.GetSwingsByPlayerAsync(playerId);
        return Ok(swings);
    }

    /// <summary>
    /// Process a swing video (test endpoint for Sprint 1)
    /// </summary>
    [HttpPost("process")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<ActionResult<object>> ProcessSwing([FromBody] ProcessSwingRequest request)
    {
        try
        {
            _logger.LogInformation("Processing swing video: {VideoPath}", request.VideoPath);

            // Call Python CV service
            var ballResult = await _pythonBridge.TrackBallAsync(request.VideoPath);
            var batResult = await _pythonBridge.TrackBatAsync(request.VideoPath);

            return Ok(new
            {
                ball = ballResult,
                bat = batResult,
                message = "Swing processed successfully (STUB data)"
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to process swing");
            return BadRequest(new { error = ex.Message });
        }
    }
}

public record ProcessSwingRequest(string VideoPath);
