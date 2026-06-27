let scene = new THREE.Scene();
let camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
let renderer = new THREE.WebGLRenderer();

renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

camera.position.z = 50;

// 光源
let light = new THREE.PointLight(0xffffff, 1);
light.position.set(50, 50, 50);
scene.add(light);

// 鼠标拾取
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();
let points = [];

// 颜色映射函数
function getColor(v) {
    let color = new THREE.Color();
    color.setHSL(0.7 - v * 0.7, 1, 0.5);
    return color;
}

// 获取数据
fetch("/data")
.then(res => res.json())
.then(data => {

    data.points.forEach(p => {
        let geometry = new THREE.SphereGeometry(0.3);
        let material = new THREE.MeshBasicMaterial({
            color: getColor(p.v),
            transparent: true,
            opacity: 0.5   // ⭐ 50%透明度
        });

        let sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(p.x, p.y, p.z);

        // 存储数据
        sphere.userData = p;

        scene.add(sphere);
        points.push(sphere);
    });

    animate();
});

// 点击事件
window.addEventListener("click", function(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    let intersects = raycaster.intersectObjects(points);

    if (intersects.length > 0) {
        let p = intersects[0].object.userData;

        document.getElementById("info").innerHTML =
            `pitch: ${p.x.toFixed(3)}<br>
             a: ${p.y.toFixed(3)}<br>
             h: ${p.z.toFixed(3)}<br>
             T: ${p.v.toFixed(4)}`;
    }
});

// 动画循环
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}