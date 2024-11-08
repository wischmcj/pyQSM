
function call_rtc(){
     return this.dataChannel.send('{"class_name":"webapp/input", "data":"Test event"}')
}

AFRAME.registerComponent('do-something', {
  init: function () {
    var sceneEl = this.el;
  }
});
