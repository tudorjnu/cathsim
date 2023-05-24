using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Guidewire : MonoBehaviour
{
    public float length = 1.0f;
    public float thickness = 0.02f;
    public int numSegments = 10;
    public Color color = Color.gray;

    private void Start()
    {
        CreateGuidewire();
    }

    private void CreateGuidewire()
    {
        // Create the guidewire parent object
        GameObject guidewire = new GameObject("Guidewire");
        guidewire.transform.parent = transform;

        // Calculate the length of each segment
        float segmentLength = length / numSegments;

        // Create individual segments
        for (int i = 0; i < numSegments; i++)
        {
            // Create a cylinder for each segment
            GameObject segment = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            segment.transform.parent = guidewire.transform;

            // Set segment dimensions
            segment.transform.localScale = new Vector3(thickness, segmentLength * 0.5f, thickness);
            segment.transform.localPosition = new Vector3(0f, i * segmentLength + segmentLength * 0.5f, 0f);

            // Set segment material
            Renderer segmentRenderer = segment.GetComponent<Renderer>();
            segmentRenderer.material.color = color;
        }
    }
}

