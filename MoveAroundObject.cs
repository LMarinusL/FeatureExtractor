using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveAroundObject : MonoBehaviour
{
    [SerializeField]
    private float _mouseSensitivity = 3.0f;
    private float _rotationX;
    private float _rotationY;
    float distanceMin = 10f;
    float distanceMax = 400f;
    float distance = 50f;



    [SerializeField]
    private Transform _target;

    [SerializeField]
    private float _distanceFromTarget = 150f;

    void Update()
    {
        float xCorrection= 1140f;
        float yCorrection= 110f;
        float zCorrection= 644f;
        float mouseX = 0f;
        float mouseY = 0f;

            mouseX = Input.GetAxis("Mouse X") * _mouseSensitivity;
            mouseY = Input.GetAxis("Mouse Y") * _mouseSensitivity;
        
        distance = Mathf.Clamp(distance - Input.GetAxis("Mouse ScrollWheel") * 5, distanceMin, distanceMax);
        _distanceFromTarget = distance;
        if (Input.GetMouseButton(0))
        {
            _rotationX += mouseY;
            _rotationY += mouseX;
        }
        Vector3 newPosition = _target.position + new Vector3(xCorrection, yCorrection, zCorrection );
        _rotationX = Mathf.Clamp(_rotationX, -60, 60);
        transform.localEulerAngles = new Vector3(_rotationX, _rotationY, 0);
        transform.position = newPosition - transform.forward * _distanceFromTarget;
    }
}
