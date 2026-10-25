const search=document.querySelector('.search');
if(search)search.addEventListener('input',()=>{const q=search.value.trim().toLowerCase();document.querySelectorAll('[data-search]').forEach(el=>{el.hidden=!el.dataset.search.includes(q);});});
function reveal(){if(!location.hash)return;const el=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(!el)return;for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;requestAnimationFrame(()=>el.scrollIntoView({block:'start'}));}
window.addEventListener('hashchange',reveal);reveal();
