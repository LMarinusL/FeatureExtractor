using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveAroundObject : MonoBehaviour
{
    [SerializeField]
    private float _mouseSensitivity = 3.0f;
    private float _rotationX;
    private float _rotationY;
    public float distanceMin = 10f;
    public float distanceMax = 100f;
    public float distance = 50f;



    [SerializeField]
    private Transform _target;

    [SerializeField]
    private float _distanceFromTarget = 50f;

    void Update()
    {
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
        _rotationX = Mathf.Clamp(_rotationX, -60, 60);
        transform.localEulerAngles = new Vector3(_rotationX, _rotationY, 0);
        transform.position = _target.position - transform.forward * _distanceFromTarget;
    }
}
