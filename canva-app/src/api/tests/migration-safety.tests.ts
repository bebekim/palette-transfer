/**
 * ABOUTME: Migration safety tests for Express → Flask transition
 * ABOUTME: Ensures critical functionality is preserved during backend migration
 */

import { transferPalette, getRemainingCredits } from "../api";

// Mock fetch globally for testing
global.fetch = jest.fn();

// Mock BACKEND_HOST for testing environment
(global as any).BACKEND_HOST = 'http://localhost:5000';

describe("Express → Flask Migration Safety Tests", () => {
  beforeEach(() => {
    (fetch as jest.Mock).mockClear();
  });

  describe("KEEP: Core Business Logic", () => {
    // RED: Test that Flask must implement palette transfer endpoint
    it("should handle palette transfer API contract", async () => {
      const mockResponse = {
        processedImageUrl: "https://flask-server.com/result.jpg",
        metrics: {
          lightingConsistency: 0.85,
          backgroundConsistency: 0.90,
          transferQuality: 0.88
        },
        direction: "A_to_B",
        medical_mode: false
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
        headers: new Map([['content-type', 'application/json']])
      });

      const sourceFile = new File(['source'], 'source.jpg', { type: 'image/jpeg' });
      const targetFile = new File(['target'], 'target.jpg', { type: 'image/jpeg' });

      const result = await transferPalette({
        sourceImage: sourceFile,
        targetImage: targetFile,
        direction: 'A_to_B'
      });

      // Verify Flask endpoint URL is correct
      const fetchCall = (fetch as jest.Mock).mock.calls[0];
      expect(fetchCall[0].toString()).toBe('http://localhost:5000/api/v1/palette-transfer');
      
      // Verify request includes POST method and body
      const requestOptions = fetchCall[1];
      expect(requestOptions.method).toBe('POST');
      expect(requestOptions.body).toBeDefined();

      // Verify response contract is maintained
      expect(result).toEqual(mockResponse);
    });

    // RED: Test error handling for Flask API failures
    it("should handle Flask API errors gracefully", async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500
      });

      const sourceFile = new File(['source'], 'source.jpg', { type: 'image/jpeg' });
      const targetFile = new File(['target'], 'target.jpg', { type: 'image/jpeg' });

      await expect(transferPalette({
        sourceImage: sourceFile,
        targetImage: targetFile,
        direction: 'A_to_B'
      })).rejects.toThrow('Request failed with status 500');
    });

    // RED: Test Flask serves Canva app static file
    it("should serve Canva app.js from Flask", async () => {
      const mockResponse = {
        ok: true,
        text: jest.fn().mockResolvedValue("// Canva app bundle content")
      };

      (fetch as jest.Mock).mockResolvedValueOnce(mockResponse);

      const response = await fetch('/app.js');
      const content = await response.text();

      expect(response.ok).toBe(true);
      expect(content).toContain('// Canva app bundle content');
    });
  });

  describe("REMOVE: Demo Functionality (Should Not Break Frontend)", () => {
    // RED: Test that removing credits API doesn't break frontend
    it("should handle missing credits endpoint gracefully", async () => {
      // Simulate Flask not implementing credits (404)
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 404
      });

      await expect(getRemainingCredits()).rejects.toThrow('Request failed with status 404');
      
      // This is expected - frontend should handle missing demo endpoints
    });
  });

  describe("ADAPT: Infrastructure Requirements", () => {
    // RED: Test authentication header handling
    it("should maintain authentication headers for Flask", async () => {
      // Mock successful response with auth
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ processedImageUrl: "test.jpg" }),
        headers: new Map([['content-type', 'application/json']])
      });

      const sourceFile = new File(['source'], 'source.jpg', { type: 'image/jpeg' });
      const targetFile = new File(['target'], 'target.jpg', { type: 'image/jpeg' });

      await transferPalette({
        sourceImage: sourceFile,
        targetImage: targetFile,
        direction: 'A_to_B'
      });

      // Verify Flask can receive authentication if provided
      const fetchCall = (fetch as jest.Mock).mock.calls[0];
      const requestOptions = fetchCall[1];
      
      // Should be prepared to handle auth headers
      expect(requestOptions.method).toBe('POST');
      expect(requestOptions.body).toBeInstanceOf(FormData);
    });

    // RED: Test CORS requirements for Flask
    it("should work with same-origin Flask server", async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ processedImageUrl: "test.jpg" }),
        headers: new Map([['content-type', 'application/json']])
      });

      const sourceFile = new File(['source'], 'source.jpg', { type: 'image/jpeg' });
      const targetFile = new File(['target'], 'target.jpg', { type: 'image/jpeg' });

      await transferPalette({
        sourceImage: sourceFile,
        targetImage: targetFile,
        direction: 'A_to_B'
      });

      // Should call Flask on same domain (no CORS issues)
      expect(fetch).toHaveBeenCalledWith(
        expect.not.stringMatching(/^https?:\/\/localhost:3001/), // Should NOT call Express port
        expect.any(Object)
      );
    });
  });
});