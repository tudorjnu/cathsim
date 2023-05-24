using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GuidewireController : MonoBehaviour
{
    public GameObject guidewire;
    public float rotationSpeed = 5f;
    public float movementSpeed = 0.2f;

    private void Update()
    {
        // Rotate the guidewire
        if (Input.GetKey(KeyCode.RightArrow))
        {
            guidewire.transform.Rotate(Vector3.up, rotationSpeed * Time.deltaTime);
        }
        else if (Input.GetKey(KeyCode.LeftArrow))
        {
            guidewire.transform.Rotate(Vector3.up, -rotationSpeed * Time.deltaTime);
        }

        // Move the guidewire forward or backward
        if (Input.GetKey(KeyCode.UpArrow))
        {
            guidewire.transform.Translate(Vector3.forward * movementSpeed * Time.deltaTime);
        }
        else if (Input.GetKey(KeyCode.DownArrow))
        {
            guidewire.transform.Translate(-Vector3.forward * movementSpeed * Time.deltaTime);
        }
    }
}

