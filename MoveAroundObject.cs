using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveAroundObject : MonoBehaviour
{
    [SerializeField]
    private float _mouseSensitivity = 3.0f;
    private float _rotationX;
    private float _rotationY;

    [SerializeField]
    private Transform _target;

    [SerializeField]
    private float _distanceFromTarget = 13.0f;

    void Update()
    {
        float mouseX = Input.GetAxis("Mouse X") * _mouseSensitivity;
        float mouseY = Input.GetAxis("Mouse Y") * _mouseSensitivity;
        _rotationX += mouseY;
        _rotationY += mouseX;
        _rotationX = Mathf.Clamp(_rotationX, -60, 60);
        transform.localEulerAngles = new Vector3(_rotationX, _rotationY, 0);
        transform.position = _target.position - transform.forward * _distanceFromTarget;
    }
}
